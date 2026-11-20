"""Index FAISS pra-hitung dan pencarian hybrid BM25 + FAISS.

Indexer berjalan lewat scripts/build_index.py, BUKAN saat Streamlit dibuka.
Dokumen hanya diambil melalui allowlist crawler web_knowledge.py.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from google.genai import types
from rank_bm25 import BM25Okapi

import web_knowledge

SCHEMA_VERSION = 1
EMBEDDING_DIMENSION = 768
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EMBED_ITEMS_PER_MINUTE = 60
DEFAULT_MAX_QUOTA_RETRIES = 3
DEFAULT_INDEX_DIR = web_knowledge.DEFAULT_CACHE_DIR / "precomputed"
CHUNK_SIZE = 950
CHUNK_OVERLAP = 100
MIN_SEMANTIC_SCORE = 0.35
STOP_WORDS = {
    "apa", "apakah", "bagaimana", "berapa", "dari", "dan", "atau", "yang", "untuk",
    "dengan", "dalam", "pada", "ini", "itu", "saya", "kami", "bisa", "dapat",
    "ada", "di", "ke", "se", "oleh", "adalah", "tentang", "lebih", "telkom",
    "university", "language", "center", "informasi", "mohon", "tolong",
}


def tokenize(value: str) -> list[str]:
    return [
        token for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) >= 3 and token not in STOP_WORDS
    ]


def recursive_chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
                         overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Potong per paragraf dan kalimat, mempertahankan sedikit konteks."""
    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

    def split(segment: str, level: int = 0) -> list[str]:
        segment = segment.strip()
        if not segment:
            return []
        if len(segment) <= chunk_size:
            return [segment]
        if level >= len(separators):
            results = []
            step = max(1, chunk_size - overlap)
            for start in range(0, len(segment), step):
                chunk = segment[start:start + chunk_size].strip()
                if chunk:
                    results.append(chunk)
                if start + chunk_size >= len(segment):
                    break
            return results
        sep = separators[level]
        parts = [part.strip() for part in segment.split(sep) if part.strip()]
        if len(parts) <= 1:
            return split(segment, level + 1)
        groups, current = [], ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            if current:
                groups.extend(split(current, level + 1))
            current = part
        if current:
            groups.extend(split(current, level + 1))
        if overlap <= 0 or len(groups) <= 1:
            return groups
        previous, result = "", []
        for group in groups:
            combined = (previous + " " + group).strip() if previous else group
            result.append(combined[:chunk_size + overlap])
            previous = group[-overlap:]
        return result

    return split(text)


def make_chunks(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks = []
    for page in pages:
        for number, text in enumerate(recursive_chunk_text(page["text"]), start=1):
            chunks.append({
                "id": len(chunks),
                "url": page["url"],
                "title": page["title"],
                "topic": page["topic"],
                "campus": page["campus"],
                "checked_at": page["checked_at"],
                "chunk_no": number,
                "text": text,
            })
    return chunks


def _embedding_config(model: str, task: str) -> types.EmbedContentConfig:
    if model.startswith("gemini-embedding-2"):
        # On Embedding 2, task_type is not sent to preserve compatibility.
        return types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION)
    return types.EmbedContentConfig(
        task_type=task, output_dimensionality=EMBEDDING_DIMENSION
    )


def embed_texts(client: Any, texts: list[str], model: str, task: str) -> list[list[float]]:
    """Satu permintaan berisi beberapa dokumen terpisah.

    Embedding 2 perlu daftar types.Content, bukan daftar string yang bisa
    digabung oleh API menjadi satu embedding.
    """
    if not texts:
        return []
    contents = (
        [types.Content(parts=[types.Part(text=text)]) for text in texts]
        if model.startswith("gemini-embedding-2") else texts
    )
    response = client.models.embed_content(
        model=model, contents=contents, config=_embedding_config(model, task)
    )
    vectors = [entry.values for entry in (response.embeddings or [])]
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Jumlah embedding tidak cocok: {len(vectors)} untuk {len(texts)} dokumen"
        )
    for vector in vectors:
        if len(vector) != EMBEDDING_DIMENSION or not np.isfinite(vector).all():
            raise RuntimeError("Dimensi/nilai embedding tidak valid")
    return [list(map(float, vector)) for vector in vectors]


def _quota_retry_delay(exc: Exception) -> float | None:
    """Extract RetryInfo from formatted Gemini exception without SDK internals."""
    message = str(exc)
    for pattern in (
        r"Please retry in\s+(\d+(?:\.\d+)?)s",
        r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s",
    ):
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _is_rate_limit(exc: Exception) -> bool:
    for attr in ("code", "status_code"):
        try:
            if int(getattr(exc, attr, 0)) == 429:
                return True
        except (ValueError, TypeError):
            pass
    return "429" in str(exc) and "RESOURCE_EXHAUSTED" in str(exc)


def _is_daily_quota(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "per day" in text or "perday" in text or
        "requests per day" in text or "daily limit" in text
    ) and _quota_retry_delay(exc) is None


def embed_with_quota_retry(
    client: Any, texts: list[str], model: str, task: str,
    max_retries: int = DEFAULT_MAX_QUOTA_RETRIES,
) -> list[list[float]]:
    """Respect RetryInfo for per-minute 429; preserve successful cache batches."""
    if max_retries < 0 or max_retries > 10:
        raise ValueError("max_retries harus 0–10")
    for attempt in range(max_retries + 1):
        try:
            return embed_texts(client, texts, model, task)
        except Exception as exc:
            if not _is_rate_limit(exc):
                raise
            if _is_daily_quota(exc):
                raise RuntimeError(
                    "Kuota embedding harian Gemini telah habis; cache yang berhasil "
                    "tetap tersimpan. Periksa kuota harian proyek di Google AI Studio."
                ) from exc
            if attempt >= max_retries:
                raise RuntimeError(
                    "Batas embedding Gemini masih terlampaui setelah retry; "
                    "cache yang berhasil tetap tersimpan. Kurangi --embed-rpm "
                    "atau lanjutkan indexing setelah kuota tersedia."
                ) from exc
            retry_info = _quota_retry_delay(exc)
            delay = max(retry_info if retry_info is not None
                        else min(15 * (2 ** attempt), 180), 1) + 2
            logging.warning(
                "Gemini 429 (RPM); retry %d/%d setelah jeda %.1f detik.",
                attempt + 1, max_retries, delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def cached_document_embeddings(
    client: Any, chunks: list[dict], model: str, cache_dir: Path,
    batch_size: int = 8,
    embed_rpm: int = DEFAULT_EMBED_ITEMS_PER_MINUTE,
    max_quota_retries: int = DEFAULT_MAX_QUOTA_RETRIES,
) -> np.ndarray:
    """Throttle berdasarkan jumlah TEKS (bukan hanya HTTP call); save per batch."""
    if not 1 <= batch_size <= 50:
        raise ValueError("batch_size harus 1–50")
    if not batch_size <= embed_rpm <= 600:
        raise ValueError("embed_rpm harus >= batch_size dan <= 600")
    if not chunks:
        raise ValueError("Tidak ada potongan dokumen yang akan diindeks")
    cache_key = f"{model}@{EMBEDDING_DIMENSION}"
    cache = web_knowledge.load_embedding_cache(cache_dir, cache_key)
    keys = [hashlib.sha256(item["text"].encode("utf-8")).hexdigest() for item in chunks]
    missing = {}
    for key, item in zip(keys, chunks):
        vector = cache.get(key)
        if vector is None or len(vector) != EMBEDDING_DIMENSION:
            missing.setdefault(key, item["text"])
    pending = list(missing.items())
    logging.info(
        "Embedding cache: %d/%d teks unik tersedia, %d belum di-embedding.",
        len(set(keys)) - len(pending), len(set(keys)), len(pending),
    )
    next_request_at = 0.0
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        remaining = next_request_at - time.monotonic()
        if remaining > 0:
            logging.info("Pacing: jeda %.1f detik untuk limit embedding.", remaining)
            time.sleep(remaining)

        vectors = embed_with_quota_retry(
            client, [text for _, text in batch], model, "RETRIEVAL_DOCUMENT",
            max_retries=max_quota_retries,
        )
        # batchEmbedContents can count each item toward the per-minute quota.
        next_request_at = time.monotonic() + len(batch) * 60.0 / embed_rpm
        for (digest, _), vector in zip(batch, vectors):
            cache[digest] = vector
        web_knowledge.save_embedding_cache(cache_dir, cache_key, cache)
        logging.info(
            "Embedding berhasil dan tersimpan: %d/%d teks baru.",
            min(start + batch_size, len(pending)), len(pending),
        )
    return normalized(np.asarray([cache[key] for key in keys], dtype="float32"))

def normalized(matrix: np.ndarray) -> np.ndarray:
    vectors = np.ascontiguousarray(matrix, dtype="float32")
    faiss.normalize_L2(vectors)
    return vectors


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_precomputed_index(client: Any, sources_path: Path,
                            index_dir: Path = DEFAULT_INDEX_DIR,
                            model: str = DEFAULT_EMBEDDING_MODEL,
                            refresh_hours: float = 12,
                            batch_size: int = 8, force: bool = False,
                            embed_rpm: int = DEFAULT_EMBED_ITEMS_PER_MINUTE,
                            max_quota_retries: int = DEFAULT_MAX_QUOTA_RETRIES) -> dict:
    sources_path = Path(sources_path).resolve()
    sources = web_knowledge.load_sources(sources_path)
    pages, report = web_knowledge.sync_sources(
        sources, cache_dir=web_knowledge.DEFAULT_CACHE_DIR,
        refresh_hours=refresh_hours, force=force,
    )
    chunks = make_chunks(pages)
    if not chunks:
        raise ValueError("Tidak ada teks resmi untuk indeks")
    vectors = cached_document_embeddings(
        client, chunks, model, web_knowledge.DEFAULT_CACHE_DIR, batch_size,
        embed_rpm=embed_rpm, max_quota_retries=max_quota_retries,
    )
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_dir = Path(index_dir).resolve()
    snapshot_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(4)
    snapshot_dir = index_dir / "snapshots" / snapshot_name
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    faiss.write_index(index, str(snapshot_dir / "index.faiss"))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "sources_config_sha256": _digest(sources_path),
        "embedding_model": model,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "total_chunks": len(chunks),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "report": report,
        "sources": sources,
    }
    web_knowledge.atomic_json(snapshot_dir / "chunks.json", {"chunks": chunks})
    web_knowledge.atomic_json(snapshot_dir / "metadata.json", metadata)
    # Pointer ditulis terakhir; snapshot sebelumnya tetap valid ketika build gagal.
    web_knowledge.atomic_json(index_dir / "current.json", {"snapshot": snapshot_name})
    return metadata


def load_precomputed_index(sources_path: Path, index_dir: Path = DEFAULT_INDEX_DIR):
    """Hanya memuat file indeks; tidak crawl dan tidak meminta embedding."""
    index_dir = Path(index_dir).resolve()
    pointer_path = index_dir / "current.json"
    if not pointer_path.exists():
        raise FileNotFoundError(
            f"Indeks belum tersedia di {index_dir}. Jalankan: "
            "python scripts/build_index.py sebelum streamlit run app.py"
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    name = pointer.get("snapshot", "")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError("Pointer indeks tidak valid")
    snapshot = index_dir / "snapshots" / name
    metadata = json.loads((snapshot / "metadata.json").read_text(encoding="utf-8"))
    chunks = json.loads((snapshot / "chunks.json").read_text(encoding="utf-8"))["chunks"]
    index = faiss.read_index(str(snapshot / "index.faiss"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Versi format indeks sudah berubah; buat ulang indeks")
    if metadata.get("sources_config_sha256") != _digest(Path(sources_path)):
        raise ValueError("sources.json berubah sejak indeks dibuat; jalankan ulang build_index.py")
    if metadata.get("total_chunks") != len(chunks) or index.ntotal != len(chunks):
        raise ValueError("Indeks FAISS tidak konsisten dengan metadata; buat ulang indeks")
    if index.d != metadata.get("embedding_dimension"):
        raise ValueError("Dimensi indeks FAISS tidak cocok")
    return index, chunks, metadata, BM25Okapi([
        tokenize(f"{c['title']} {c['topic']} {c['text']}") or ["_empty_"]
        for c in chunks
    ])


def retrieve_hybrid(client: Any, index: faiss.Index, chunks: list[dict],
                    bm25: BM25Okapi, model: str, question: str,
                    top_k: int = 6) -> tuple[list[dict], str | None]:
    """Reciprocal Rank Fusion, dengan syarat bukti lexical/semantic.

    Jika embedding pertanyaan gagal, BM25 tetap bisa dipakai; statusnya dicatat
    agar pengguna tidak mengira pencarian semantik berhasil.
    """
    qtokens = tokenize(question)
    if not qtokens or not chunks:
        return [], None
    lexical_scores = bm25.get_scores(qtokens)
    # BM25Okapi can assign zero IDF on very small corpora. Preserve
    # literal non-stopword overlap as a safe lexical fallback.
    overlaps = [
        len(set(qtokens).intersection(tokenize(
            f"{item['title']} {item['topic']} {item['text']}"
        )))
        for item in chunks
    ]
    lexical_rank = sorted(
        (i for i in range(len(chunks)) if overlaps[i] > 0),
        key=lambda i: (float(lexical_scores[i]), overlaps[i]),
        reverse=True,
    )[:20]
    semantic_rank = []
    semantic_scores = {}
    warning = None
    try:
        vector = normalized(np.asarray(embed_texts(
            client, [question], model, "RETRIEVAL_QUERY"
        ), dtype="float32"))
        scores, indices = index.search(vector, min(20, len(chunks)))
        for score, i in zip(scores[0], indices[0]):
            if i >= 0 and float(score) >= MIN_SEMANTIC_SCORE:
                semantic_rank.append(int(i))
                semantic_scores[int(i)] = float(score)
    except Exception:
        logging.exception("Query embedding failed: BM25 fallback")
        warning = "Pencarian semantik tidak tersedia; hasil memakai pencarian kata kunci."

    rrf = {}
    for ranks in (lexical_rank, semantic_rank):
        for position, i in enumerate(ranks, 1):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + position)
    order = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:top_k]
    results = []
    for i in order:
        item = dict(chunks[i])
        item["score"] = float(semantic_scores.get(i, 0.0))
        item["retrieval_score"] = rrf[i]
        item["retrieval"] = (
            "hybrid" if i in lexical_rank and i in semantic_rank else
            "faiss" if i in semantic_rank else "bm25"
        )
        results.append(item)
    return results, warning
