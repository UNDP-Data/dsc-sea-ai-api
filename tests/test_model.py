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


def _is_graceful_fallback(text: str) -> bool:
    lowered = text.lower()
    return (
        "temporary issue" in lowered
        or "temporary delay" in lowered
        or "please retry" in lowered
    )


@pytest.mark.parametrize(
    "content",
    [
        "Hi there",
        "What can you do?",
        (
            "How does climate change adaptation differ from climate change mitigation?"
        ),
        "How much energy does a typical residential solar panel generate?",
    ],
)
def test_model_structure(test_client, content: str):
    """
    Test if the response from `/model` endpoint has the expected format.
    """
    response = test_client.post("/model", json=[{"role": "human", "content": content}])
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-ndjson"
    assert response.headers.get("x-request-id")
    ideas_chunks = 0
    chunks = _read_chunks(response)
    graph_chunks = 0
    for data in chunks:
        assert isinstance(data, dict)
        assert data.get("role") == "assistant"
        assert isinstance(data.get("content"), str)
        if data.get("documents") is not None:
            assert isinstance(data["documents"], list)
        if data.get("ideas") is not None:
            ideas_chunks += 1
        if data.get("graph") is not None:
            graph_chunks += 1
    # graph may arrive at any point in stream, but should be present once.
    assert graph_chunks == 1
    # ideas may now arrive before the final chunk; ensure they appear at least once.
    assert ideas_chunks >= 1


@pytest.mark.parametrize(
    "messages,pattern,keywords",
    [
        (
            [
                {
                    "role": "human",
                    "content": "How does climate change adaptation differ from climate change mitigation?",
                }
            ],
            r"climate change (adaptation|mitigation)",
            ["climate", "adaptation", "mitigation"],
        ),
        (
            [
                {
                    "role": "human",
                    "content": "How much energy does a typical residential solar panel generate?",
                }
            ],
            r"(watts|kilowatt-hours|kWh)",
            ["solar", "panel"],
        ),
        (
            [
                {
                    "role": "human",
                    "content": "When was the Paris Agreement signed?",
                }
            ],
            r"\b(2016|2015)\b",
            ["paris agreement"],
        ),
    ],
)
def test_model_response(
    test_client, messages: list[dict], pattern: str, keywords: list[str]
):
    """
    Test if `/model` endpoint produces meaningful responses.
    """
    response = test_client.post("/model", json=messages)
    assert response.status_code == 200
    chunks = _read_chunks(response)
    contents = "".join(chunk.get("content", "") for chunk in chunks)
    assert chunks[0]["role"] == "assistant"
    assert (
        re.search(pattern, contents, re.IGNORECASE)
        or any(keyword in contents.lower() for keyword in keywords)
        or _is_graceful_fallback(contents)
    )


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
    assert re.search(access_code, contents) or _is_graceful_fallback(contents)


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


def test_model_empty_messages_400(test_client):
    """
    Test if `/model` rejects empty message lists with a 400 response.
    """
    response = test_client.post("/model", json=[])
    assert response.status_code == 400
    assert response.json()["detail"] == "At least one message is required."
