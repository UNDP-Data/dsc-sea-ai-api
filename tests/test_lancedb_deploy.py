from __future__ import annotations

import pandas as pd
import pytest

from src.rag_system import lancedb_deploy


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


class FakeTable:
    def __init__(self, rows: int):
        self.rows = rows

    async def count_rows(self):
        return self.rows


class FakeConnection:
    def __init__(self):
        self.calls = []

    async def create_table(self, table_name, data, mode="create"):
        self.calls.append((table_name, data.copy(), mode))
        return FakeTable(len(data))


def test_embed_dataframe_texts_adds_vector_column():
    frame = pd.DataFrame({"content": ["short", "longer text"]})

    result = lancedb_deploy.embed_dataframe_texts(
        frame,
        embedder=FakeEmbedder(),
        text_column="content",
        progress=None,
    )

    assert "vector" in result.columns
    assert result["vector"].tolist() == [[5.0, 1.0], [11.0, 1.0]]
    assert "vector" not in frame.columns


def test_embed_dataframe_texts_requires_text_column():
    with pytest.raises(ValueError, match="Missing text column"):
        lancedb_deploy.embed_dataframe_texts(
            pd.DataFrame({"body": ["text"]}),
            embedder=FakeEmbedder(),
            text_column="content",
            progress=None,
        )


@pytest.mark.asyncio
async def test_deploy_dataframe_table_uses_connection_create_table():
    connection = FakeConnection()
    frame = pd.DataFrame({"content": ["a", "b"], "vector": [[1.0], [2.0]]})

    result = await lancedb_deploy.deploy_dataframe_table(
        connection,
        "sample_chunks",
        frame,
        mode="overwrite",
        progress=None,
    )

    assert result.table_name == "sample_chunks"
    assert result.rows == 2
    assert result.columns == ["content", "vector"]
    assert connection.calls[0][0] == "sample_chunks"
    assert connection.calls[0][2] == "overwrite"


@pytest.mark.asyncio
async def test_deploy_dataframe_table_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="empty dataframe"):
        await lancedb_deploy.deploy_dataframe_table(
            FakeConnection(),
            "empty",
            pd.DataFrame(),
            progress=None,
        )
