"""
Functions for interacting with GenAI models via Azure OpenAI.
"""

import os
import pkgutil

import yaml
from openai import AzureOpenAI
from pydantic import BaseModel

from . import genai
from .entities import Document, Message

__all__ = ["get_client", "generate_response", "embed_text"]

PROMPTS = yaml.safe_load(pkgutil.get_data(__name__, "prompts.yaml"))


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


def extract_entities(user_query: str) -> list[str]:

    class ResponseFormat(BaseModel):
        entities: list[str]

    response: ResponseFormat = genai.generate_response(
        prompt=user_query,
        system_message=PROMPTS["extract_entities"],
        response_format=ResponseFormat,
    )
    return response.entities


def get_answer(
    user_question: str, documents: list[Document], messages: list[Message]
) -> str:
    response = genai.generate_response(
        prompt=user_question,
        system_message=PROMPTS["answer_question"].format(
            documents=documents, messages=messages
        ),
        temperature=0.3,
        top_p=0.8,
        frequency_penalty=0.6,
        presence_penalty=0.8,
    )
    return response


def generate_query_ideas(user_query: str) -> list[str]:

    class ResponseFormat(BaseModel):
        ideas: list[str]

    response: ResponseFormat = genai.generate_response(
        prompt=user_query,
        system_message=PROMPTS["suggest_ideas"],
        response_format=ResponseFormat,
    )
    return response.ideas
