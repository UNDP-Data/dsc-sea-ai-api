"""Reusable LanceDB deployment helpers for notebook-driven RAG pipelines."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class TableDeploymentResult:
    """Summary for a LanceDB table write."""

    table_name: str
    mode: str
    rows: int
    columns: list[str]

    def model_dump(self) -> dict[str, Any]:
        """Return a JSON-serializable result mapping."""
        return asdict(self)


def _progress(message: str, callback: ProgressCallback | None) -> None:
    if callback:
        callback(message)


def dataframe_from_records(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Build a dataframe from record mappings with a clearer empty-input error."""
    rows = list(records)
    if not rows:
        raise ValueError("Cannot build a LanceDB table from zero records.")
    return pd.DataFrame(rows)


def write_parquet_artifact(
    dataframe: pd.DataFrame,
    path: str | Path,
    *,
    compression: str = "gzip",
    progress: ProgressCallback | None = print,
) -> Path:
    """Write a dataframe artifact and return the resolved output path."""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _progress(f"[lancedb-deploy] Writing parquet artifact: {output_path}", progress)
    dataframe.to_parquet(output_path, compression=compression)
    return output_path


def embed_dataframe_texts(
    dataframe: pd.DataFrame,
    *,
    embedder: Any,
    text_column: str,
    vector_column: str = "vector",
    batch_size: int = 128,
    progress: ProgressCallback | None = print,
) -> pd.DataFrame:
    """
    Add embeddings for a dataframe text column.

    The embedder must expose the LangChain-compatible `embed_documents(texts)`
    method used by the existing Azure OpenAI embedding client.
    """
    if text_column not in dataframe.columns:
        raise ValueError(f"Missing text column: {text_column}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    output = dataframe.copy()
    texts = output[text_column].fillna("").astype(str).tolist()
    embeddings: list[Any] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        _progress(f"[lancedb-deploy] Embedding rows {start + 1}-{end} of {total}", progress)
        embeddings.extend(embedder.embed_documents(texts[start:end]))
    output[vector_column] = embeddings
    return output


async def deploy_dataframe_table(
    connection: Any,
    table_name: str,
    dataframe: pd.DataFrame,
    *,
    mode: str = "overwrite",
    progress: ProgressCallback | None = print,
) -> TableDeploymentResult:
    """Create or overwrite a LanceDB table from a dataframe."""
    if dataframe.empty:
        raise ValueError(f"Refusing to deploy empty dataframe to table {table_name!r}.")
    _progress(
        f"[lancedb-deploy] Writing table {table_name!r} mode={mode!r} rows={len(dataframe)}",
        progress,
    )
    table = await connection.create_table(table_name, data=dataframe, mode=mode)
    rows = await table.count_rows()
    result = TableDeploymentResult(
        table_name=table_name,
        mode=mode,
        rows=int(rows),
        columns=[str(column) for column in dataframe.columns],
    )
    _progress(f"[lancedb-deploy] Table {table_name!r} rows={result.rows}", progress)
    return result


def deploy_dataframe_table_sync(
    connection: Any,
    table_name: str,
    dataframe: pd.DataFrame,
    *,
    mode: str = "overwrite",
    progress: ProgressCallback | None = print,
) -> TableDeploymentResult:
    """Synchronous wrapper for scripts that are not already inside an event loop."""
    return asyncio.run(
        deploy_dataframe_table(
            connection,
            table_name,
            dataframe,
            mode=mode,
            progress=progress,
        )
    )


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a dataframe artifact by file extension."""
    input_path = Path(path).expanduser()
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix == ".jsonl":
        return pd.read_json(input_path, lines=True)
    if suffix == ".json":
        return pd.read_json(input_path)
    raise ValueError(
        f"Unsupported dataframe input extension {suffix!r}. "
        "Use .parquet, .csv, .json, or .jsonl."
    )
