#!/usr/bin/env python3
"""Deploy a dataframe artifact to a LanceDB table.

This is the script form of the notebook deployment helper. It expects the
publication parsing/chunking pipeline to have already produced a dataframe file
such as `.parquet`, `.csv`, `.json`, or `.jsonl`.
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

from src import database, genai  # noqa: E402
from src.rag_system import get_profile, lancedb_deploy  # noqa: E402


def _progress(message: str) -> None:
    print(message, flush=True)


async def _run(args: argparse.Namespace) -> dict:
    profile = get_profile(args.assistant_id) if args.assistant_id else None
    dataframe = lancedb_deploy.load_dataframe(args.input)
    if args.embed_text_column:
        embedder = genai.get_embedding_client()
        dataframe = lancedb_deploy.embed_dataframe_texts(
            dataframe,
            embedder=embedder,
            text_column=args.embed_text_column,
            vector_column=args.vector_column,
            batch_size=args.batch_size,
            progress=_progress,
        )
        if args.embedded_output:
            lancedb_deploy.write_parquet_artifact(
                dataframe,
                args.embedded_output,
                compression=args.compression,
                progress=_progress,
            )

    connection = await database.get_connection(profile=profile)
    try:
        result = await lancedb_deploy.deploy_dataframe_table(
            connection,
            args.table,
            dataframe,
            mode=args.mode,
            progress=_progress,
        )
        return result.model_dump()
    finally:
        close = getattr(connection, "close", None)
        if callable(close):
            maybe_coro = close()
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input dataframe artifact: .parquet, .csv, .json, or .jsonl.")
    parser.add_argument("--table", required=True, help="Target LanceDB table name.")
    parser.add_argument(
        "--assistant-id",
        default="",
        help="Optional assistant profile id for profile-specific LanceDB storage.",
    )
    parser.add_argument(
        "--mode",
        default="overwrite",
        choices=["overwrite", "create"],
        help="LanceDB create_table mode. Default: overwrite.",
    )
    parser.add_argument(
        "--embed-text-column",
        default="",
        help="Optional text column to embed before deployment.",
    )
    parser.add_argument(
        "--vector-column",
        default="vector",
        help="Vector column name when --embed-text-column is used.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--embedded-output",
        default="",
        help="Optional parquet path to save the dataframe after embeddings are added.",
    )
    parser.add_argument("--compression", default="gzip", help="Parquet compression for --embedded-output.")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(_run(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
