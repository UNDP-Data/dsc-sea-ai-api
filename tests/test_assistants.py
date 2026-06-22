"""Tests for route-prefixed RAG assistants."""

import json

from langchain_core.messages import AIMessageChunk

import main as main_module
from src.entities import Chunk, Document


def _read_chunks(response) -> list[dict]:
    return [json.loads(line) for line in response.iter_lines() if line]


def test_list_assistants_includes_configured_profiles(test_client):
    response = test_client.get("/assistants")
    assert response.status_code == 200
    payload = response.json()
    assistant_ids = {item["assistant_id"] for item in payload}
    assert {"sea", "sample"}.issubset(assistant_ids)


def test_unknown_assistant_model_returns_404(test_client):
    response = test_client.post(
        "/assistants/unknown/model",
        json=[{"role": "human", "content": "hello"}],
    )
    assert response.status_code == 404


def test_assistant_model_uses_profile_prompts_and_omits_graph(test_client, monkeypatch):
    seen_system_messages = []

    async def fake_stream_chat_response(*, system_message, **_kwargs):
        seen_system_messages.append(system_message)
        if "configured publication corpus" in system_message:
            raise AssertionError("non-default assistants should wait for publication retrieval")
        elif "publication excerpts" in system_message:
            yield AIMessageChunk(content="Sample grounded answer.")

    async def fake_generate_query_ideas(_messages, profile=None):
        assert profile is not None
        assert profile.assistant_id == "sample"
        return ["Sample follow-up?"]

    class FakeClient:
        def __init__(self, _connection, *, profile=None, table_names=None):
            self.profile = profile
            self.table_names = table_names or getattr(profile, "table_names", {})
            self.connection = _connection

        async def retrieve_chunks(self, _query):
            assert self.profile.assistant_id == "sample"
            return (
                [
                    Chunk(
                        document_id="sample-doc",
                        title="Sample Food Systems Policy Note",
                        year=2026,
                        language="en",
                        url="https://example.org/sample-food-systems-policy-note",
                        summary="Sample summary.",
                        content="Sample publication evidence.",
                    )
                ],
                [
                    Document(
                        document_id="sample-doc",
                        title="Sample Food Systems Policy Note",
                        year=2026,
                        language="en",
                        url="https://example.org/sample-food-systems-policy-note",
                        summary="Sample summary.",
                    )
                ],
            )

    class FakeConnection:
        def close(self):
            return None

    async def fake_get_connection(*_args, **_kwargs):
        return FakeConnection()

    monkeypatch.setattr(main_module.genai, "stream_chat_response", fake_stream_chat_response)
    monkeypatch.setattr(main_module.genai, "generate_query_ideas", fake_generate_query_ideas)
    monkeypatch.setattr(main_module.database, "Client", FakeClient)
    monkeypatch.setattr(main_module.database, "get_connection", fake_get_connection)

    response = test_client.post(
        "/assistants/sample/model",
        json=[{"role": "human", "content": "How do food systems relate to water management?"}],
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-ndjson"
    chunks = _read_chunks(response)
    assert chunks
    assert all(chunk.get("graph") is None for chunk in chunks)
    assert any(chunk.get("documents") for chunk in chunks)
    combined = "".join(chunk.get("content", "") for chunk in chunks)
    assert "Sample grounded answer" in combined
    assert not any("configured publication corpus" in message for message in seen_system_messages)
    assert any("publication excerpts" in message for message in seen_system_messages)


def test_sample_assistant_blocks_out_of_profile_query(test_client):
    response = test_client.post(
        "/assistants/sample/model",
        json=[{"role": "human", "content": "How should I visit Madrid?"}],
    )
    assert response.status_code == 200
    chunks = _read_chunks(response)
    combined = "".join(chunk.get("content", "") for chunk in chunks)
    assert "sample publication corpus" in combined.lower()
    assert all(chunk.get("graph") is None for chunk in chunks)
