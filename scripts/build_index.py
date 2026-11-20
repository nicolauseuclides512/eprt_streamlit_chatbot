"""Offline/pre-deploy index builder. Does not import or run Streamlit UI.

Windows:
  .\.venv\Scripts\python.exe scripts\build_index.py
Then:
  .\.venv\Scripts\python.exe -m streamlit run app.py

On Streamlit Cloud, provide a persistent/prebuilt snapshot. Local .cache files
are not guaranteed to survive a rebuild; do not assume Cloud already has index.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gemini_client import create_gemini_client
from rag_pipeline import (
    DEFAULT_EMBEDDING_MODEL, DEFAULT_INDEX_DIR,
    build_precomputed_index,
)
import web_knowledge


def local_config() -> dict:
    path = ROOT / ".streamlit" / "secrets.toml"
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def main() -> None:
    secrets = local_config()
    parser = argparse.ArgumentParser(description="Precompute website LaC FAISS index")
    parser.add_argument("--sources", default=os.getenv(
        "SOURCES_CONFIG_PATH", secrets.get("SOURCES_CONFIG_PATH", str(web_knowledge.DEFAULT_SOURCES_FILE))
    ))
    parser.add_argument("--output", default=os.getenv(
        "PRECOMPUTED_INDEX_DIR", secrets.get("PRECOMPUTED_INDEX_DIR", str(DEFAULT_INDEX_DIR))
    ))
    parser.add_argument("--model", default=os.getenv(
        "EMBEDDING_MODEL", secrets.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    ))
    parser.add_argument("--refresh-hours", type=float, default=float(os.getenv(
        "SOURCE_SYNC_HOURS", secrets.get("SOURCE_SYNC_HOURS", "12")
    )))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force-refresh", action="store_true", help="Fetch all sources again")
    args = parser.parse_args()

    key = os.getenv("GEMINI_API_KEY") or secrets.get("GEMINI_API_KEY")
    if not key or key == "ISI_API_KEY_ANDA":
        raise SystemExit("GEMINI_API_KEY belum diisi pada .streamlit/secrets.toml atau environment")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    client = create_gemini_client(key)
    metadata = build_precomputed_index(
        client=client,
        sources_path=Path(args.sources),
        index_dir=Path(args.output),
        model=args.model,
        refresh_hours=args.refresh_hours,
        batch_size=args.batch_size,
        force=args.force_refresh,
    )
    print(f"Index READY: {metadata['total_chunks']} chunks, "
          f"{metadata['report']['available_sources']} official sources")
    print(f"Model: {metadata['embedding_model']} ({metadata['embedding_dimension']} dims)")
    print(f"Indexed at: {metadata['built_at']}")
    print(f"Stored in: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
