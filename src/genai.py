"""
Functions for interacting with GenAI models via Azure OpenAI.
"""

import os

from openai import AzureOpenAI
from pydantic import BaseModel

__all__ = ["get_client", "generate_response", "embed_text"]


def get_client():
    client = AzureOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
    )
    return client


def generate_response(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    **kwargs,
) -> str | BaseModel:
    # use the defaults if no kwargs are provided
    params = {"temperature": 0} | kwargs
    client = get_client()
    response = client.beta.chat.completions.parse(
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


def embed_text(text: str) -> list[float]:
    client = get_client()
    response = client.embeddings.create(model=os.environ["EMBED_MODEL"], input=text)
    return response.data[0].embedding
