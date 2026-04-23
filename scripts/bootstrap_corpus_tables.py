#!/usr/bin/env python3
"""
Bootstrap canonical `sources` and `documents` tables from the existing `chunks` table.

Optionally rewrites the `chunks` table so every chunk includes `document_id` and
basic provenance fields required by the document-centric retrieval flow.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src import database


def _progress(message: str) -> None:
    print(f"[bootstrap] {message}", flush=True)


async def _run(args) -> dict[str, int]:
    _progress("Opening database connection...")
    connection = await database.get_connection()
    try:
        client = database.Client(connection)
        return await client.bootstrap_corpus_tables(
            overwrite=args.overwrite,
            rewrite_chunks=args.rewrite_chunks,
            parser_version=args.parser_version,
            embedding_version=args.embedding_version,
            status=args.status,
            progress=_progress,
        )
    finally:
        _progress("Closing database connection...")
        close = getattr(connection, "close", None)
        if callable(close):
            result = close()
            if asyncio.iscoroutine(result):
                await result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sources/documents tables instead of keeping current content.",
    )
    parser.add_argument(
        "--rewrite-chunks",
        action="store_true",
        help="Overwrite the chunks table with enriched rows that include document provenance.",
    )
    parser.add_argument(
        "--parser-version",
        default="bootstrap-v1",
        help="Parser version stamp to write into document metadata.",
    )
    parser.add_argument(
        "--embedding-version",
        default="legacy",
        help="Embedding version stamp to write into document metadata.",
    )
    parser.add_argument(
        "--status",
        default="approved",
        help="Document approval status to assign to bootstrapped document records.",
    )
    args = parser.parse_args()
    _progress("Starting corpus bootstrap...")
    if not args.rewrite_chunks:
        _progress(
            "Running in metadata-only mode. The chunks table will NOT be rewritten; "
            "chunk provenance fields such as document_id/chunk_id will remain unavailable."
        )
    result = asyncio.run(_run(args))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
