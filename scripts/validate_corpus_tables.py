#!/usr/bin/env python3
"""
Validate corpus metadata completeness for `documents`, `sources`, and enriched `chunks`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src import database


REQUIRED_DOCUMENT_FIELDS = (
    "document_id",
    "source_id",
    "canonical_title",
    "url",
    "language",
    "document_type",
    "year",
    "summary",
    "status",
)


async def _maybe_open(client: database.Client, name: str):
    return await client.open_optional_table(name)


async def _run() -> dict:
    print("[validate] Opening database connection...", flush=True)
    connection = await database.get_connection()
    try:
        client = database.Client(connection)
        print("[validate] Opening tables...", flush=True)
        documents_table = await _maybe_open(client, "documents")
        sources_table = await _maybe_open(client, "sources")
        chunks_table = await _maybe_open(client, "chunks")

        result: dict = {
            "sources": {"exists": bool(sources_table), "count": 0},
            "documents": {"exists": bool(documents_table), "count": 0},
            "chunks": {"exists": bool(chunks_table), "count": 0},
        }

        if sources_table is not None:
            print("[validate] Counting sources...", flush=True)
            result["sources"]["count"] = await sources_table.count_rows()

        if documents_table is not None:
            print("[validate] Loading documents metadata...", flush=True)
            rows = await documents_table.query().to_list()
            result["documents"]["count"] = len(rows)
            missing_counts = Counter()
            status_counts = Counter((row.get("status") or "unknown") for row in rows)
            for row in rows:
                for field in REQUIRED_DOCUMENT_FIELDS:
                    value = row.get(field)
                    if value in (None, "", [], {}):
                        missing_counts[field] += 1
            result["documents"]["status_counts"] = dict(status_counts)
            result["documents"]["missing_required"] = dict(missing_counts)

        if chunks_table is not None:
            print("[validate] Sampling chunk provenance...", flush=True)
            schema = await chunks_table.schema()
            field_names = {field.name for field in schema}
            result["chunks"]["count"] = await chunks_table.count_rows()
            result["chunks"]["field_names"] = sorted(field_names)
            provenance_fields = ["document_id", "chunk_id", "chunk_index"]
            result["chunks"]["provenance_ready"] = all(
                field in field_names for field in provenance_fields
            )
            if result["chunks"]["provenance_ready"]:
                rows = await (
                    chunks_table.query()
                    .select(provenance_fields)
                    .limit(5000)
                    .to_list()
                )
                result["chunks"]["sample_missing_document_id"] = sum(
                    1 for row in rows if not row.get("document_id")
                )
                result["chunks"]["sample_missing_chunk_id"] = sum(
                    1 for row in rows if not row.get("chunk_id")
                )
            else:
                result["chunks"]["sample_missing_document_id"] = None
                result["chunks"]["sample_missing_chunk_id"] = None

        return result
    finally:
        print("[validate] Closing database connection...", flush=True)
        close = getattr(connection, "close", None)
        if callable(close):
            result = close()
            if asyncio.iscoroutine(result):
                await result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(asyncio.run(_run()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
