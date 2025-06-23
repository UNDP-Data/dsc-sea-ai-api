"""
Functions for interacting with GenAI models via Azure OpenAI.
"""

import os
import pkgutil
from typing import AsyncGenerator

import yaml
from langchain_core.messages import (
    AIMessageChunk,
    MessageLikeRepresentation,
    SystemMessage,
)
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import BaseModel

from .entities import AssistantResponse, Document, Message

__all__ = [
    "get_chat_client",
    "get_embedding_client",
    "generate_response",
    "stream_response",
]

PROMPTS = yaml.safe_load(pkgutil.get_data(__name__, "prompts.yaml"))


def get_chat_client(
    temperature: float = 0.0, timeout: int = 10, **kwargs
) -> AzureChatOpenAI:
    """
    Get a chat client for Azure OpenAI service.

    Parameters
    ----------
    temperature : float, default=0.0
        Model temperature setting.
    timeout : int, default=10
        Request timeout setting in seconds.
    **kwargs
        Additional keyword arguments to pass to `AzureChatOpenAI`.

    Returns
    -------
    AzureChatOpenAI
       An Azure OpenAI integration client for chat models.
    """
    return AzureChatOpenAI(
        azure_deployment=os.environ["CHAT_MODEL"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-12-01-preview",
        api_key=os.environ["AZURE_OPENAI_KEY"],
        temperature=temperature,
        timeout=timeout,
        **kwargs,
    )


def get_embedding_client() -> AzureOpenAIEmbeddings:
    """
    Get an embedding client for Azure OpenAI service.

    Returns
    -------
    AzureOpenAIEmbeddings
        An Azure OpenAI integration client for embedding.
    """
    return AzureOpenAIEmbeddings(
        model=os.environ["EMBED_MODEL"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
    )


async def generate_response(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    schema: BaseModel | None = None,
    **kwargs,
) -> str | BaseModel:
    """
    Generate a response using Azure OpenAI service.

    This function supports structured outputs via `response_format` kwarg.

    Parameters
    ----------
    prompt : str
        User message.
    system_message : str, optional
        System message to customise model behaviour.
    schema : BaseModel, optional
        `pydantic` schema for structured output.
    **kwargs
        Addtional keyword arguments to pass to `get_chat_client`.

    Returns
    -------
    str or BaseModel
        String if no `response_format` is specified, otherwise a Pydantic model.
    """
    chat = get_chat_client(**kwargs)
    if schema is not None:
        chat = chat.with_structured_output(schema)
    response = await chat.ainvoke(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
    )
    return response if schema is not None else response.content


async def stream_response(
    messages: MessageLikeRepresentation, **kwargs
) -> AsyncGenerator[str, None]:
    """
    Stream a response using Azure OpenAI service.

    Parameters
    ----------
    messages : MessageLikeRepresentation
        Model input as accepted by `astream` method.
    **kwargs
        Addtional keyword arguments to pass to `get_chat_client`.

    Yields
    ------
    str
        Chunk content.
    """
    chat = get_chat_client(**kwargs)
    async for chunk in chat.astream(messages):
        if isinstance(chunk, AIMessageChunk):
            yield chunk.content


async def extract_entities(user_query: str) -> list[str]:
    """
    Extract relevant entities from the user query.

    Parameters
    ----------
    user_query : str
        Raw user message.

    Returns
    -------
    list[str]
        List of entities extracted from the user message.
    """

    class ResponseFormat(BaseModel):
        """
        Response format for leveraging structured outputs.
        """

        entities: list[str]

    response: ResponseFormat = await generate_response(
        prompt=user_query,
        system_message=PROMPTS["extract_entities"],
        schema=ResponseFormat,
    )
    return response.entities


async def get_answer(
    messages: list[Message],
    documents: list[Document],
    response: AssistantResponse,
) -> AsyncGenerator[str, None]:
    """
    Respond to the user message using RAG and conversation history.

    Parameters
    ----------
    messages : list[Message]
        Conversation history as a list of messages.
    documents : list[Document]
        List of relevant documents to ground the answer in.
    messages : list[Message]
        Conversation history as a list of messages.

    Yields
    ------
    str
        String representation of JSON model response.
    """
    messages = [
        SystemMessage(PROMPTS["answer_question"].format(documents=documents))
    ] + [message.to_langchain() for message in messages]
    async for chunk in stream_response(
        messages=messages,
        temperature=0.3,
        top_p=0.8,
        frequency_penalty=0.6,
        presence_penalty=0.8,
    ):
        # send deltas only
        response.content = chunk
        yield response.model_dump_json() + "\n"
        # once the full response has been yielded, nullify properties to reduce payload
        response.graph, response.ideas, response.documents = None, None, None


async def generate_query_ideas(user_query: str) -> list[str]:
    """
    Generate query ideas based on the user message.

    Parameters
    ----------
    user_query : str
        Raw user message.

    Returns
    -------
    list[str]
        List of query ideas based on the user message.
    """

    class ResponseFormat(BaseModel):
        """
        Response format for leveraging structured outputs.
        """

        ideas: list[str]

    response: ResponseFormat = await generate_response(
        prompt=user_query,
        system_message=PROMPTS["suggest_ideas"],
        schema=ResponseFormat,
    )
    return response.ideas
