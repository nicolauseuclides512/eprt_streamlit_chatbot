"""Pengambilan dan sinkronisasi sumber web publik untuk chatbot LaC.

Hanya URL HTTPS yang secara eksplisit ada dalam konfigurasi dan
hostname yang ditinjau dalam ALLOWED_HOSTS yang dapat diakses.
Tidak mengikuti tautan secara otomatis dan tidak mengakses halaman login.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
import trafilatura
from bs4 import BeautifulSoup

APP_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCES_FILE = APP_DIR / "sources.json"
DEFAULT_CACHE_DIR = APP_DIR / ".cache" / "web_rag"
ALLOWED_HOSTS = frozenset({"lac.telkomuniversity.ac.id"})
USER_AGENT = "LaC-RAG-ResearchBot/0.1"
HTTP_TIMEOUT = 15
MAX_HTML_BYTES = 2_000_000
MAX_SOURCES = 20
RETRY_AFTER_SECONDS = 0.4
LIVE_QUERY_PATTERN = re.compile(
    r"\b(jadwal|biaya|harga|tarif|pendaftaran|kuota|sesi|tanggal|"
    r"kapan|hari ini|bulan ini|jam|terbaru|202[5-9])\b",
    flags=re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_url(url: str) -> str:
    if not isinstance(url, str):
        raise ValueError("URL sumber harus berupa teks")
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED_HOSTS
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"URL tidak termasuk sumber HTTPS resmi yang diizinkan: {url}")
    return url


def load_sources(config_path: Path = DEFAULT_SOURCES_FILE) -> list[dict]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    raw_sources = config.get("sources", [])
    if not isinstance(raw_sources, list) or not 1 <= len(raw_sources) <= MAX_SOURCES:
        raise ValueError("Daftar sources harus memuat 1 sampai 20 URL")
    sources = []
    seen = set()
    for entry in raw_sources:
        if not isinstance(entry, dict):
            raise ValueError("Format sumber web tidak valid")
        if entry.get("enabled", True) is False:
            continue
        url = validate_url(entry.get("url"))
        if url in seen:
            raise ValueError(f"Duplikasi URL sumber: {url}")
        seen.add(url)
        sources.append({
            "url": url,
            "title": str(entry.get("title") or url).strip(),
            "topic": str(entry.get("topic") or "Layanan umum").strip(),
            "campus": str(entry.get("campus") or "Lintas kampus").strip(),
            "dynamic": bool(entry.get("dynamic", False)),
        })
    if not sources:
        raise ValueError("Tidak ada sumber aktif")
    return sources


def atomic_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as tmp:
            temp_name = tmp.name
            json.dump(content, tmp, ensure_ascii=False)
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def load_snapshot(cache_dir: Path = DEFAULT_CACHE_DIR) -> dict:
    path = Path(cache_dir) / "pages.json"
    if not path.exists():
        return {"pages": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data.get("pages"), dict):
            return data
    except (OSError, ValueError, TypeError):
        pass
    return {"pages": {}}


def _robots_allowed(session: requests.Session, url: str, robots_cache: dict) -> bool:
    host = urlparse(url).hostname
    if host not in robots_cache:
        robots_url = f"https://{host}/robots.txt"
        response = session.get(
            robots_url,
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
            allow_redirects=False,
        )
        try:
            if response.status_code in (404, 410):
                robots_cache[host] = None
            else:
                # Jika akses robots.txt tidak jelas, jangan crawl secara otomatis.
                response.raise_for_status()
                if response.status_code in (301, 302, 303, 307, 308):
                    raise RuntimeError("robots.txt dialihkan; perlu pemeriksaan manual")
                policy = RobotFileParser()
                policy.parse(response.text.splitlines())
                robots_cache[host] = policy
        finally:
            response.close()
    policy = robots_cache[host]
    return policy is None or policy.can_fetch(USER_AGENT, url)


def _safe_get_html(session: requests.Session, url: str) -> str:
    current = validate_url(url)
    for _ in range(4):
        response = session.get(
            current,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
            timeout=HTTP_TIMEOUT,
            allow_redirects=False,
            stream=True,
        )
        try:
            if response.status_code in (301, 302, 303, 307, 308):
                destination = response.headers.get("Location")
                if not destination:
                    raise ValueError("Redirect tanpa tujuan")
                current = validate_url(urljoin(current, destination))
                continue
            response.raise_for_status()
            if "text/html" not in response.headers.get("Content-Type", "").lower():
                raise ValueError("Sumber tidak menyediakan HTML")
            chunks = []
            size = 0
            for piece in response.iter_content(chunk_size=65536):
                size += len(piece)
                if size > MAX_HTML_BYTES:
                    raise ValueError("Ukuran HTML melebihi batas")
                chunks.append(piece)
            raw = b"".join(chunks)
            return raw.decode(response.encoding or "utf-8", errors="replace")
        finally:
            response.close()
    raise ValueError("Terlalu banyak redirect")


def extract_page(html_source: str) -> tuple[str, str]:
    soup = BeautifulSoup(html_source, "html.parser")
    heading = soup.find("h1")
    title = heading.get_text(" ", strip=True) if heading else ""
    if not title and soup.title:
        title = soup.title.get_text(" ", strip=True)
    text = trafilatura.extract(
        html_source,
        include_tables=True,
        include_comments=False,
        favor_recall=True,
    )
    if not text:
        for selector in ("script", "style", "nav", "footer", "header", "form"):
            for element in soup.select(selector):
                element.decompose()
        text = soup.get_text("\n", strip=True)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < 80:
        raise ValueError("Teks halaman terlalu pendek untuk knowledge base")
    return title, text


def fetch_page(
    source: dict,
    session: requests.Session,
    robots_cache: dict,
) -> dict:
    url = validate_url(source["url"])
    if not _robots_allowed(session, url, robots_cache):
        raise PermissionError(f"robots.txt tidak mengizinkan pengambilan {url}")
    html_source = _safe_get_html(session, url)
    extracted_title, text = extract_page(html_source)
    return {
        **source,
        "title": extracted_title or source["title"],
        "text": text,
        "digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "checked_at": utc_now(),
    }


def _due(checked_at: str | None, refresh_hours: float) -> bool:
    if not checked_at:
        return True
    try:
        last = datetime.fromisoformat(checked_at)
        if last.tzinfo is None:
            return True
        return (datetime.now(timezone.utc) - last).total_seconds() >= refresh_hours * 3600
    except (TypeError, ValueError):
        return True


def sync_sources(
    sources: list[dict],
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh_hours: float = 12,
    force: bool = False,
    session: requests.Session | None = None,
) -> tuple[list[dict], dict]:
    """Simpan hasil pengambilan sukses; kegagalan tidak menghapus snapshot lama."""
    cache_dir = Path(cache_dir)
    previous = load_snapshot(cache_dir).get("pages", {})
    updated = {}
    errors = []
    changed = 0
    checked = 0
    owns_session = session is None
    session = session or requests.Session()
    robots_cache = {}
    try:
        for source in sources:
            url = source["url"]
            prior = previous.get(url)
            if prior and not force and not _due(prior.get("checked_at"), refresh_hours):
                updated[url] = {**prior, **{key: value for key, value in source.items() if key != "title"}}
                continue
            try:
                fresh = fetch_page(source, session, robots_cache)
                checked += 1
                if prior and prior.get("digest") == fresh["digest"]:
                    fresh["changed_at"] = prior.get("changed_at", fresh["checked_at"])
                else:
                    fresh["changed_at"] = fresh["checked_at"]
                    changed += 1
                updated[url] = fresh
            except (requests.RequestException, ValueError, RuntimeError, PermissionError) as exc:
                errors.append(f"{source['title']}: {type(exc).__name__}")
                if prior:
                    updated[url] = {**prior, **{key: value for key, value in source.items() if key != "title"}}
            if checked:
                time.sleep(RETRY_AFTER_SECONDS)
    finally:
        if owns_session:
            session.close()
    if not updated:
        raise RuntimeError("Belum ada sumber web yang berhasil diambil. Periksa jaringan dan robots.txt.")
    atomic_json(cache_dir / "pages.json", {"pages": updated})
    pages = [updated[item["url"]] for item in sources if item["url"] in updated]
    report = {
        "total_sources": len(sources),
        "available_sources": len(pages),
        "checked": checked,
        "changed": changed,
        "errors": errors,
        "oldest_check": min((page["checked_at"] for page in pages), default=None),
        "latest_check": max((page["checked_at"] for page in pages), default=None),
    }
    return pages, report


def live_check(question: str, contexts: list[dict], sources: list[dict]) -> tuple[dict | None, str | None]:
    """Satu pengecekan online ketika pertanyaan mengenai informasi dinamis."""
    if not LIVE_QUERY_PATTERN.search(question):
        return None, None
    by_url = {source["url"]: source for source in sources if source["dynamic"]}
    source = next((by_url[ctx["url"]] for ctx in contexts if ctx["url"] in by_url), None)
    if source is None:
        return None, None
    try:
        with requests.Session() as session:
            record = fetch_page(source, session, {})
        record["text"] = record["text"][:7000]
        return record, None
    except (requests.RequestException, ValueError, RuntimeError, PermissionError):
        return None, "Verifikasi langsung belum berhasil; periksa halaman resmi untuk jadwal/biaya terbaru."


def load_embedding_cache(cache_dir: Path, model: str) -> dict:
    path = Path(cache_dir) / "embeddings.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("model") == model and isinstance(data.get("vectors"), dict):
            return data["vectors"]
    except (OSError, ValueError, TypeError):
        pass
    return {}


def save_embedding_cache(cache_dir: Path, model: str, vectors: dict) -> None:
    atomic_json(Path(cache_dir) / "embeddings.json", {"model": model, "vectors": vectors})
