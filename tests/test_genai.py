"""
Basic tests for functions in `genai` module.
"""

import json

import pytest
from langchain_core.messages import AIMessageChunk
from pydantic import BaseModel

from src import genai
from src.entities import AssistantResponse, Message


@pytest.mark.parametrize(
    "text", ["sustainable development", "solar energy", "decarbonisation"]
)
@pytest.mark.asyncio
async def test_embed_text(text: str):
    """
    Test if `embed_text` function produces expected results.
    """
    embedder = genai.get_embedding_client()
    embedding = await embedder.aembed_query(text)
    assert len(embedding) == 1_024
    assert abs(sum(embedding)) > 0.0


@pytest.mark.asyncio
async def test_generate_response():
    """
    Test if `generate_response` function produces expected results.
    """
    system_message = "Extract all dates from a text."
    text = """UNDP is based on the merging of the United Nations Expanded Programme of Technical Assistance,
    created in 1949, and the United Nations Special Fund, established in 1958."""

    # base case
    response = await genai.generate_response(text, system_message)
    assert isinstance(response, str)
    assert "1949" in response and "1958" in response

    # with structured outputs
    class Response(BaseModel):
        years: list[int]

    response = await genai.generate_response(text, system_message, schema=Response)
    assert isinstance(response, Response)
    assert [1949, 1958] == response.years


@pytest.mark.asyncio
async def test_get_answer_emits_ideas_before_final_chunk(monkeypatch):
    """
    Test if `get_answer` can emit ideas during token streaming when ready early.
    """

    async def fake_stream_response(*_args, **_kwargs):
        yield AIMessageChunk(content="first ")
        yield AIMessageChunk(content="second")

    async def fake_generate_query_ideas(_messages):
        return ["Idea A", "Idea B"]

    monkeypatch.setattr(genai, "stream_response", fake_stream_response)
    monkeypatch.setattr(genai, "generate_query_ideas", fake_generate_query_ideas)

    response = AssistantResponse(role="assistant", content="")
    messages = [Message(role="human", content="test query")]
    chunks = []
    async for payload in genai.get_answer(messages, response, tools=[]):
        chunks.append(json.loads(payload))

    assert len(chunks) >= 3
    # At least one non-final chunk should include ideas if they are ready early.
    assert any(chunk.get("ideas") for chunk in chunks[:-1])
    # Final chunk remains compatible.
    assert chunks[-1].get("ideas") == ["Idea A", "Idea B"]


@pytest.mark.asyncio
async def test_get_answer_continues_when_idea_generation_fails(monkeypatch):
    """
    Test if `get_answer` still streams answer content when ideas generation fails.
    """

    async def fake_stream_response(*_args, **_kwargs):
        yield AIMessageChunk(content="content ")
        yield AIMessageChunk(content="continues")

    async def failing_generate_query_ideas(_messages):
        raise RuntimeError("ideas-failure")

    monkeypatch.setattr(genai, "stream_response", fake_stream_response)
    monkeypatch.setattr(genai, "generate_query_ideas", failing_generate_query_ideas)

    response = AssistantResponse(role="assistant", content="")
    messages = [Message(role="human", content="test query")]
    chunks = []
    async for payload in genai.get_answer(messages, response, tools=[]):
        chunks.append(json.loads(payload))

    combined = "".join(chunk.get("content", "") for chunk in chunks)
    assert "content continues" in combined
    # Final chunk should still be emitted with no ideas.
    assert chunks[-1].get("ideas") is None
