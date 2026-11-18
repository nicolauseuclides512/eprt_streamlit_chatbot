"""Pengujian unit crawler dan snapshot tanpa koneksi jaringan."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import web_knowledge as wk

URL = "https://lac.telkomuniversity.ac.id/layanan/tes-bahasa/eprt/"
SOURCE = {"url": URL, "title": "EPrT", "topic": "Tes", "campus": "Lintas kampus", "dynamic": True}


class FakeResponse:
    def __init__(self, status_code=200, headers=None, text="", data=b""):
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "text/html"}
        self.text = text
        self.data = data
        self.encoding = "utf-8"
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise ValueError("HTTP error")

    def iter_content(self, chunk_size=65536):
        yield self.data

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.urls = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        return self.responses.pop(0)


class WebKnowledgeTests(unittest.TestCase):
    def test_url_allowlist(self):
        self.assertEqual(wk.validate_url(URL), URL)
        for url in (
            "http://lac.telkomuniversity.ac.id/",
            "https://127.0.0.1/",
            "https://lac.telkomuniversity.ac.id.evil.example/",
            "https://evil.example@lac.telkomuniversity.ac.id/",
            "https://lac.telkomuniversity.ac.id:8443/",
            "https://lac.telkomuniversity.ac.id/?x=1",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                wk.validate_url(url)

    def test_load_sources_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.json"
            path.write_text(json.dumps({"sources": [SOURCE]}), encoding="utf-8")
            self.assertEqual(wk.load_sources(path)[0]["url"], URL)
            path.write_text(json.dumps({"sources": [SOURCE, SOURCE]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                wk.load_sources(path)

    def test_redirect_outside_allowlist_blocked(self):
        response = FakeResponse(status_code=302, headers={"Location": "https://evil.example/private"})
        session = FakeSession([response])
        with self.assertRaises(ValueError):
            wk._safe_get_html(session, URL)
        self.assertTrue(response.closed)
        self.assertEqual(len(session.urls), 1)

    def test_robots_disallow(self):
        session = FakeSession([FakeResponse(text="User-agent: *\nDisallow: /layanan/\n")])
        self.assertFalse(wk._robots_allowed(session, URL, {}))

    def test_robots_not_found_allows(self):
        session = FakeSession([FakeResponse(status_code=404)])
        self.assertTrue(wk._robots_allowed(session, URL, {}))

    def test_extract_html_text_and_title(self):
        article = "Informasi EPrT Telkom University Language Center. " * 9
        markup = "<html><body><main><h1>EPrT</h1><article><p>" + article + "</p></article></main></body></html>"
        title, text = wk.extract_page(markup)
        self.assertEqual(title, "EPrT")
        self.assertIn("Informasi EPrT", text)

    def test_sync_retains_last_good_snapshot(self):
        record = {**SOURCE, "text": "informasi " * 40, "digest": "123",
                  "checked_at": wk.utc_now(), "changed_at": wk.utc_now()}
        with tempfile.TemporaryDirectory() as directory, patch("web_knowledge.time.sleep"):
            with patch("web_knowledge.fetch_page", return_value=record):
                pages, report = wk.sync_sources([SOURCE], cache_dir=Path(directory), refresh_hours=0)
            self.assertEqual(report["changed"], 1)
            self.assertEqual(len(pages), 1)
            with patch("web_knowledge.fetch_page", side_effect=ValueError("offline")):
                pages2, report2 = wk.sync_sources([SOURCE], cache_dir=Path(directory), refresh_hours=0)
            self.assertEqual(pages2[0]["text"], record["text"])
            self.assertEqual(len(report2["errors"]), 1)

    def test_same_digest_does_not_count_as_change(self):
        record = {**SOURCE, "text": "isi " * 40, "digest": "same",
                  "checked_at": wk.utc_now()}
        with tempfile.TemporaryDirectory() as directory, patch("web_knowledge.time.sleep"):
            with patch("web_knowledge.fetch_page", return_value=record):
                _, first = wk.sync_sources([SOURCE], cache_dir=Path(directory), refresh_hours=0)
                _, second = wk.sync_sources([SOURCE], cache_dir=Path(directory), refresh_hours=0)
            self.assertEqual(first["changed"], 1)
            self.assertEqual(second["changed"], 0)

    def test_live_only_dynamic_question(self):
        with patch("web_knowledge.fetch_page") as fetch:
            found, warning = wk.live_check("Jelaskan pengertian EPrT", [{"url": URL}], [SOURCE])
        self.assertIsNone(found)
        self.assertIsNone(warning)
        fetch.assert_not_called()

    def test_embedding_cache_respects_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            wk.save_embedding_cache(path, "test-model", {"hash": [0.2, 0.5]})
            self.assertEqual(wk.load_embedding_cache(path, "test-model")["hash"], [0.2, 0.5])
            self.assertEqual(wk.load_embedding_cache(path, "other-model"), {})

    def test_corrupted_snapshot_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "pages.json").write_text("{", encoding="utf-8")
            self.assertEqual(wk.load_snapshot(Path(directory))["pages"], {})


if __name__ == "__main__":
    unittest.main()
