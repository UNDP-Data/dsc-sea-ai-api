"""
Basic tests for `/model` endpoint.
"""

import json
import re
from random import choices
from string import ascii_letters, digits

import pytest


def _read_chunks(response) -> list[dict]:
    chunks = [json.loads(line) for line in response.iter_lines() if line]
    assert chunks
    return chunks


def _get_graph_chunk(chunks: list[dict]) -> dict | None:
    return next((chunk for chunk in chunks if chunk.get("graph") is not None), None)


@pytest.mark.parametrize(
    "content,rag",
    [
        ("Hi there", False),
        ("What can you do?", False),
        (
            "How does climate change adaptation differ from climate change mitigation?",
            False,
        ),
        ("How much energy does a typical residential solar panel generate?", True),
    ],
)
def test_model_structure(test_client, content: str, rag: bool):
    """
    Test if the response from `/model` endpoint has the expected format.
    """
    response = test_client.post("/model", json=[{"role": "human", "content": content}])
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-ndjson"
    assert response.headers.get("x-request-id")
    documents = []
    ideas_chunks = 0
    chunks = _read_chunks(response)
    graph_chunks = 0
    for data in chunks:
        assert isinstance(data, dict)
        assert data.get("role") == "assistant"
        assert isinstance(data.get("content"), str)
        documents.append(data["documents"] is not None)
        if data.get("ideas") is not None:
            ideas_chunks += 1
        if data.get("graph") is not None:
            graph_chunks += 1
    # graph may arrive at any point in stream, but should be present once.
    assert graph_chunks == 1
    # ideas may now arrive before the final chunk; ensure they appear at least once.
    assert ideas_chunks >= 1
    if rag:
        # at least one chunk contains documents
        assert any(documents)
    else:
        # none of the chunks contain documents
        assert not any(documents)


@pytest.mark.parametrize(
    "messages,pattern",
    [
        (
            [
                {
                    "role": "human",
                    "content": "How does climate change adaptation differ from climate change mitigation?",
                }
            ],
            r"climate change (adaptation|mitigation)",
        ),
        (
            [
                {
                    "role": "human",
                    "content": "How much energy does a typical residential solar panel generate?",
                }
            ],
            r"(watts|kilowatt-hours|kWh)",
        ),
        (
            [
                {
                    "role": "human",
                    "content": "When was the Paris Agreement signed?",
                }
            ],
            r"\b(2016|2015)\b",
        ),
    ],
)
def test_model_response(test_client, messages: list[dict], pattern: str):
    """
    Test if `/model` endpoint produces meaningful responses.
    """
    response = test_client.post("/model", json=messages)
    assert response.status_code == 200
    chunks = _read_chunks(response)
    contents = "".join(chunk.get("content", "") for chunk in chunks)
    assert chunks[0]["role"] == "assistant"
    assert re.search(pattern, contents, re.IGNORECASE)


def test_model_memory(test_client):
    """
    Test if `/model` endpoint correctly utilises message history.
    """
    access_code = "".join(choices(digits + ascii_letters, k=16))
    messages = [
        {
            "role": "human",
            "content": f"Hi there. I am Bob (Access code: {access_code})!",
        },
        {
            "role": "assistant",
            "content": "Hello Bob! It's great to meet you."
            " How can I assist you today in your journey through the Sustainable Energy Academy?",
        },
        {
            "role": "human",
            "content": "Could you remind me my name and access code?",
        },
    ]
    response = test_client.post("/model", json=messages)
    assert response.status_code == 200
    chunks = _read_chunks(response)
    contents = "".join(chunk.get("content", "") for chunk in chunks)
    assert chunks[0]["role"] == "assistant"
    assert re.search(access_code, contents)


def test_model_graph_version_v2(test_client):
    """
    Test if `/model` supports returning V2 graph schema.
    """
    response = test_client.post(
        "/model",
        params={"graph_version": "v2"},
        json=[{"role": "human", "content": "climate change mitigation"}],
    )
    assert response.status_code == 200
    chunks = _read_chunks(response)
    graph_chunk = _get_graph_chunk(chunks)
    assert graph_chunk is not None
    graph_data = graph_chunk["graph"]
    assert "nodes" in graph_data and "edges" in graph_data
    if graph_data["nodes"]:
        node = graph_data["nodes"][0]
        assert "tier" in node
        assert "neighbourhood" not in node


def test_model_graph_version_v1(test_client):
    """
    Test if `/model` can still return V1 graph schema for compatibility.
    """
    response = test_client.post(
        "/model",
        params={"graph_version": "v1"},
        json=[{"role": "human", "content": "climate change mitigation"}],
    )
    assert response.status_code == 200
    chunks = _read_chunks(response)
    graph_chunk = _get_graph_chunk(chunks)
    assert graph_chunk is not None
    graph_data = graph_chunk["graph"]
    assert "nodes" in graph_data and "edges" in graph_data
    if graph_data["nodes"]:
        node = graph_data["nodes"][0]
        assert "neighbourhood" in node
        assert "tier" not in node
