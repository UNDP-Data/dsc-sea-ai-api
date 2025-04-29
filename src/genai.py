"""
Functions for interacting with GenAI models via Azure OpenAI.
"""

import os

from openai import AzureOpenAI

__all__ = ["get_client", "generate_response", "embed_text"]


def get_client():
    client = AzureOpenAI(
        api_version="2024-07-01-preview",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )
    return client


def generate_response(prompt):
    client = get_client()
    response = client.chat.completions.create(
        model=os.environ["CHAT_MODEL"],
        temperature=0,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def embed_text(text: str) -> list[float]:
    client = get_client()
    response = client.embeddings.create(model=os.environ["EMBED_MODEL"], input=text)
    return response.data[0].embedding
