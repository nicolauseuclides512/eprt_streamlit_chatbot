"""Offline tests for FAISS snapshots, Gemini embedding batching and hybrid BM25."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import rag_pipeline as rag


def vector(first: float) -> list[float]:
    values = [0.0] * rag.EMBEDDING_DIMENSION
    values[0] = first
    values[1] = 1.0
    return values


class EmbeddingBatchTests(unittest.TestCase):
    def test_embedding2_uses_distinct_contents(self):
        calls = []
        class FakeModels:
            def embed_content(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(embeddings=[
                    SimpleNamespace(values=vector(0.1)),
                    SimpleNamespace(values=vector(0.2)),
                ])
        client = SimpleNamespace(models=FakeModels())
        result = rag.embed_texts(client, ["one", "two"], "gemini-embedding-2", "RETRIEVAL_DOCUMENT")
        self.assertEqual(len(result), 2)
        self.assertEqual(len(calls[0]["contents"]), 2)
        self.assertEqual(calls[0]["contents"][0].parts[0].text, "one")

    def test_refuses_aggregated_embedding_response(self):
        client = SimpleNamespace(models=SimpleNamespace(
            embed_content=lambda **kw: SimpleNamespace(
                embeddings=[SimpleNamespace(values=vector(0.2))]
            )
        ))
        with self.assertRaisesRegex(RuntimeError, "Jumlah embedding tidak cocok"):
            rag.embed_texts(client, ["one", "two"], "gemini-embedding-2", "RETRIEVAL_DOCUMENT")

    def test_checkpoint_after_each_successful_batch(self):
        chunks = [{"text": f"Dokumen unik {i}"} for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            with patch("rag_pipeline.time.sleep"), patch("rag_pipeline.embed_texts", side_effect=[
                [vector(0.2)], RuntimeError("API interrupted")
            ]):
                with self.assertRaisesRegex(RuntimeError, "API interrupted"):
                    rag.cached_document_embeddings(
                        object(), chunks, "test-embedding", Path(tmp), batch_size=1
                    )
            cache = rag.web_knowledge.load_embedding_cache(
                Path(tmp), f"test-embedding@{rag.EMBEDDING_DIMENSION}"
            )
            self.assertEqual(len(cache), 1)



class QuotaHandlingTests(unittest.TestCase):
    class FakeQuotaError(Exception):
        code = 429

    def test_parse_retry_info_fractional_seconds(self):
        error = self.FakeQuotaError(
            "429 RESOURCE_EXHAUSTED. Please retry in 51.893622939s. "
            "'retryDelay': '51s'"
        )
        self.assertAlmostEqual(rag._quota_retry_delay(error), 51.893622939)

    def test_parse_retry_delay_field(self):
        error = self.FakeQuotaError("429: {'retryDelay': '51s'}")
        self.assertEqual(rag._quota_retry_delay(error), 51.0)

    def test_429_recovers_after_server_backoff(self):
        error = self.FakeQuotaError(
            "429 RESOURCE_EXHAUSTED. Please retry in 51s."
        )
        with patch("rag_pipeline.embed_texts", side_effect=[
            error, [vector(0.2)]
        ]) as embed, patch("rag_pipeline.time.sleep") as sleep:
            result = rag.embed_with_quota_retry(
                object(), ["test"], "model", "RETRIEVAL_DOCUMENT",
                max_retries=2,
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(embed.call_count, 2)
        sleep.assert_called_once_with(53.0)

    def test_daily_limit_not_retried(self):
        error = self.FakeQuotaError(
            "429 RESOURCE_EXHAUSTED: free tier requests per day limit exceeded"
        )
        with (
            patch("rag_pipeline.embed_texts", side_effect=error),
            patch("rag_pipeline.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "harian"):
                rag.embed_with_quota_retry(
                    object(), ["test"], "model", "RETRIEVAL_DOCUMENT"
                )
            sleep.assert_not_called()

    def test_other_error_is_not_retried(self):
        with (
            patch("rag_pipeline.embed_texts", side_effect=ValueError("invalid model")),
            patch("rag_pipeline.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(ValueError, "invalid model"):
                rag.embed_with_quota_retry(
                    object(), ["test"], "model", "RETRIEVAL_DOCUMENT"
                )
            sleep.assert_not_called()

    def test_429_retry_limit_gives_resumable_message(self):
        error = self.FakeQuotaError("429 RESOURCE_EXHAUSTED. Please retry in 1s.")
        with patch("rag_pipeline.embed_texts", side_effect=error), (
            patch("rag_pipeline.time.sleep") as sleep
        ):
            with self.assertRaisesRegex(RuntimeError, "cache"):
                rag.embed_with_quota_retry(
                    object(), ["test"], "model", "RETRIEVAL_DOCUMENT",
                    max_retries=1,
                )
            self.assertEqual(sleep.call_count, 1)

    def test_proactive_pacing_counts_text_items(self):
        chunks = [{"text": f"Cache miss unique text {i}"} for i in range(2)]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("rag_pipeline.embed_texts", return_value=[vector(0.2)]),
            patch("rag_pipeline.time.monotonic", return_value=0),
            patch("rag_pipeline.time.sleep") as sleep,
        ):
            rag.cached_document_embeddings(
                object(), chunks, "mock", Path(tmp),
                batch_size=1, embed_rpm=60,
            )
            sleep.assert_called_once_with(1.0)

    def test_cache_saved_before_following_batch_hits_429(self):
        chunks = [{"text": f"Document {i} unique"} for i in range(2)]
        error = self.FakeQuotaError("429 RESOURCE_EXHAUSTED. Please retry in 1s.")
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("rag_pipeline.time.sleep"),
            patch("rag_pipeline.embed_texts", side_effect=[
                [vector(0.2)], error
            ]),
        ):
            with self.assertRaisesRegex(RuntimeError, "cache"):
                rag.cached_document_embeddings(
                    object(), chunks, "mock", Path(tmp), batch_size=1,
                    max_quota_retries=0,
                )
            cache = rag.web_knowledge.load_embedding_cache(
                Path(tmp), f"mock@{rag.EMBEDDING_DIMENSION}"
            )
            self.assertEqual(len(cache), 1)


class IndexSnapshotTests(unittest.TestCase):
    def test_build_and_reload_without_live_api(self):
        url = "https://lac.telkomuniversity.ac.id/layanan/tes-bahasa/eprt/"
        page = {
            "url": url, "title": "Panduan EPrT", "topic": "EPrT",
            "campus": "Lintas kampus",
            "checked_at": "2026-11-20T03:00:00+00:00",
            "text": ("EPrT listening grammar reading dan prosedur pendaftaran. " * 10),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = root / "sources.json"
            sources.write_text(json.dumps({"sources": [{
                "url": url, "title": "EPrT", "topic": "EPrT",
                "campus": "Lintas kampus", "dynamic": True
            }]}), encoding="utf-8")
            report = {"total_sources": 1, "available_sources": 1,
                      "errors": [], "latest_check": page["checked_at"]}
            with patch.object(rag.web_knowledge, "DEFAULT_CACHE_DIR", root / "cache"), (
                patch.object(rag.web_knowledge, "sync_sources", return_value=([page], report))
            ), patch("rag_pipeline.embed_texts", side_effect=lambda client, texts, model, task:
                     [vector(i * .1) for i, _ in enumerate(texts)]):
                meta = rag.build_precomputed_index(
                    object(), sources, root / "index", model="mock-embedding",
                )
                self.assertGreaterEqual(meta["total_chunks"], 1)
                index, chunks, loaded, lexical = rag.load_precomputed_index(sources, root / "index")
                self.assertEqual(index.ntotal, len(chunks))
                self.assertEqual(loaded["embedding_model"], "mock-embedding")
                self.assertTrue(any(lexical.get_scores(rag.tokenize("EPrT")).tolist()))
                sources.write_text('{"sources":[]}', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "sources.json berubah"):
                    rag.load_precomputed_index(sources, root / "index")

    def test_missing_index_gives_actionable_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "build_index.py"):
                rag.load_precomputed_index(Path(tmp) / "sources.json", Path(tmp) / "not-built")


class HybridRetrievalTests(unittest.TestCase):
    def test_lexical_fallback_if_gemini_embedding_fails(self):
        import faiss
        from rank_bm25 import BM25Okapi
        chunks = [
            {"title": "EPrT", "topic": "Tes", "text": "EPrT dan reading grammar", "url": "official"},
            {"title": "Kursus", "topic": "Kursus", "text": "Pelatihan bahasa Indonesia", "url": "official"}
        ]
        bm25 = BM25Okapi([
            rag.tokenize(f"{item['title']} {item['topic']} {item['text']}")
            for item in chunks
        ])
        index = faiss.IndexFlatIP(rag.EMBEDDING_DIMENSION)
        index.add(np.asarray([vector(0.2), vector(0.4)], dtype="float32"))
        with patch("rag_pipeline.embed_texts", side_effect=TimeoutError("API timed out")):
            found, warning = rag.retrieve_hybrid(
                object(), index, chunks, bm25,
                "mock", "Apa yang dimaksud tes EPrT?", 2
            )
        self.assertTrue(found)
        self.assertEqual(found[0]["title"], "EPrT")
        self.assertIn("kata kunci", warning)

    def test_irrelevant_question_returns_no_source(self):
        import faiss
        from rank_bm25 import BM25Okapi
        chunks = [{"title": "Tes", "topic": "EPrT", "text": "reading grammar", "url": "x"}]
        bm25 = BM25Okapi([rag.tokenize("reading grammar EPrT")])
        index = faiss.IndexFlatIP(rag.EMBEDDING_DIMENSION)
        index.add(np.asarray([vector(0.2)], dtype="float32"))
        with patch("rag_pipeline.embed_texts", side_effect=TimeoutError("offline")):
            found, _ = rag.retrieve_hybrid(
                object(), index, chunks, bm25,
                "mock", "Bagaimana resep soto ayam?", 2
            )
        self.assertFalse(found)
