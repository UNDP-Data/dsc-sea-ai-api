"""
Basic tests for `/model` endpoint.
"""

import json
import re
from random import choices
from string import ascii_letters, digits

import pytest


@pytest.mark.parametrize(
    "content,rag",
    [
        ("Hi there", False),
        ("What can you do?", False),
        (
            "How does climate change adaptation differ from climate change mitigation?",
            True,
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
    documents = []
    for index, line in enumerate(response.iter_lines()):
        assert line
        data = json.loads(line)
        assert isinstance(data, dict)
        assert data.get("role") == "assistant"
        assert isinstance(data.get("content"), str)
        documents.append(data["documents"] is not None)
        if index == 0:
            # only the first object contains graph
            assert data["graph"] is not None
        else:
            assert data["graph"] is None
    # ideas are included in the last chunk only
    assert data["ideas"] is not None
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
    contents = []
    for index, line in enumerate(response.iter_lines()):
        if index == 0:
            data = json.loads(line)
        contents.append(json.loads(line).get("content", ""))
    data["content"] = "".join(contents)
    assert data["role"] == "assistant"
    assert re.search(pattern, data["content"], re.IGNORECASE)


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
