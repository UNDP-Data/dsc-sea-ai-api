"""
Functions for interacting with GenAI models via Azure OpenAI.
"""

import os
import pkgutil

import yaml
from openai import AsyncAzureOpenAI
from pydantic import BaseModel

from .entities import Document, Message

__all__ = ["get_client", "generate_response", "embed_text"]

PROMPTS = yaml.safe_load(pkgutil.get_data(__name__, "prompts.yaml"))


def get_client() -> AsyncAzureOpenAI:
    """
    Get a asynchronous client for Azure OpenAI service.

    Returns
    -------
    AsyncAzureOpenAI
        An asynchronous Azure OpenAI client.
    """
    client = AsyncAzureOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
    )
    return client


async def generate_response(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
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
    **kwargs
        Extra arguments to be passed to `client.beta.chat.completions.parse`.

    Returns
    -------
    str or Base Model
        String if no `response_format` is specified, otherwise a Pydantic model.
    """
    # use the defaults if no kwargs are provided
    params = {"temperature": 0} | kwargs
    client = get_client()
    response = await client.beta.chat.completions.parse(
        model=os.environ["CHAT_MODEL"],
        **params,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        timeout=10,
    )
    message = response.choices[0].message
    return message.parsed if "response_format" in params else message.content


async def embed_text(text: str) -> list[float]:
    """
    Embed a text into a multidimensional vector space.

    Parameters
    ----------
    text : str
        A text to embed. Must be shorter than 8,191 tokens.

    Returns
    -------
    list[float]
        Embedding for the text as a 1,536 vector of float.
    """
    client = get_client()
    response = await client.embeddings.create(
        model=os.environ["EMBED_MODEL"], input=text
    )
    return response.data[0].embedding


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
        response_format=ResponseFormat,
    )
    return response.entities


async def get_answer(
    user_query: str, documents: list[Document], messages: list[Message]
) -> str:
    """
    Respond to the user message using RAG and conversation history.

    Parameters
    ----------
    user_query : str
        Raw user message.
    documents : list[Document]
        List of relevant documents to ground the answer in.
    messages : list[Message]
        Conversation history as a list of messages.

    Returns
    -------
    str
        Model response.
    """
    response = await generate_response(
        prompt=user_query,
        system_message=PROMPTS["answer_question"].format(
            documents=documents, messages=messages
        ),
        temperature=0.3,
        top_p=0.8,
        frequency_penalty=0.6,
        presence_penalty=0.8,
    )
    return response


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
        response_format=ResponseFormat,
    )
    return response.ideas
