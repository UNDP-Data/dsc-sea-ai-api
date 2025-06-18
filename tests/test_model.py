"""
Basic tests for `/model` endpoint.
"""

import json
import re
from random import choices
from string import ascii_letters, digits

import pytest

from tests.utils import validate_graph


def test_model_structure(test_client):
    """
    Test if the response from `/model` endpoint has the expected format.
    """
    messages = [
        {
            "role": "human",
            "content": "How does climate change adaptation differ from climate change mitigation?",
        }
    ]
    response = test_client.post("/model", json=messages)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-ndjson"
    contents = []
    for index, line in enumerate(response.iter_lines()):
        assert line
        data = json.loads(line)
        assert isinstance(data, dict)
        assert data.get("role") == "assistant"
        assert (content := data.get("content")) is not None
        assert isinstance(content, str)
        contents.append(content)
        # only the first object contains graph, documents and ideas
        if index == 0:
            assert (documents := data.get("documents")) is not None
            assert isinstance(documents, list)
            assert len(documents) > 0
            assert all(map(lambda document: isinstance(document, dict), documents))
            validate_graph(data["graph"])
        else:
            assert data.get("ideas") is None
            assert data.get("documents") is None
            assert data.get("graph") is None
    content = "".join(contents)
    assert "climate change adaptation" in content.lower()


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
            r"\b2016\b",
        ),
    ],
)
def test_model_response(test_client, messages: list[dict], pattern: str):
    """
    Test if `/model` endpoint produces meaningful responses.
    """
    response = test_client.post("/model", json=messages)
    assert response.status_code == 200
    contents = []
    for index, line in enumerate(response.iter_lines()):
        if index == 0:
            data = json.loads(line)
        contents.append(json.loads(line).get("content", ""))
    data["content"] = "".join(contents)
    assert data["role"] == "assistant"
    assert re.search(pattern, data["content"])


@pytest.mark.parametrize(
    "name,access_code",
    [
        (name, "".join(choices(digits + ascii_letters, k=16)))
        for name in ("Bob", "VIKI", "Nebuchadnezzar")
    ],
)
def test_model_memory(test_client, name: str, access_code: str):
    """
    Test if `/model` endpoint correctly utilises message history.
    """
    messages = [
        {
            "role": "human",
            "content": f"Hi there. I am {name} (Access code: {access_code})!",
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
    contents = []
    for index, line in enumerate(response.iter_lines()):
        if index == 0:
            data = json.loads(line)
        contents.append(json.loads(line).get("content", ""))
    data["content"] = "".join(contents)
    assert data["role"] == "assistant"
    content = data["content"]
    assert re.search(name, content)
    assert re.search(access_code, content)
