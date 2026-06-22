#!/usr/bin/env python3
"""Export the canonical documents table to a CSV for expert annotation."""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path
from inspect import isawaitable

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from src import corpus, database  # noqa: E402
from src.rag_system import get_profile  # noqa: E402

FIELDS = [
    "document_id",
    "title",
    "url",
    "publisher",
    "source",
    "year",
    "publication_date",
    "document_type",
    "language",
    "summary",
    "topics",
    "geographies",
    "status",
]


def _serialize(value):
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


async def _run(output: Path, assistant_id: str) -> int:
    profile = get_profile(assistant_id)
    connection = await database.get_connection(profile=profile)
    try:
        client = database.Client(connection, profile=profile)
        table = await client.open_optional_table(client.table_name("documents"))
        if table is None:
            print(
                f"Documents table {client.table_name('documents')!r} does not exist.",
                file=sys.stderr,
            )
            return 1
        rows = await table.query().to_list()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for row in rows:
                document = corpus.document_record_to_api(row)
                writer.writerow(
                    {
                        "document_id": document.document_id or "",
                        "title": document.title,
                        "url": document.url,
                        "publisher": document.publisher or "",
                        "source": document.source or "",
                        "year": document.year or "",
                        "publication_date": document.publication_date or "",
                        "document_type": document.document_type or "",
                        "language": document.language,
                        "summary": document.summary or "",
                        "topics": _serialize(document.topics),
                        "geographies": _serialize(document.geographies),
                        "status": row.get("status", ""),
                    }
                )
        print(f"Wrote {output}")
        return 0
    finally:
        close = getattr(connection, "close", None)
        if close is not None:
            result = close()
            if isawaitable(result):
                await result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/retrieval_benchmark/corpus_inventory.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--assistant-id",
        default="sea",
        help="RAG assistant profile id to export.",
    )
    args = parser.parse_args()
    return asyncio.run(_run(Path(args.output), args.assistant_id))


if __name__ == "__main__":
    raise SystemExit(main())
