"""
Basic tests for functions in `genai` module.
"""

import pytest
from pydantic import BaseModel

from src import genai


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
