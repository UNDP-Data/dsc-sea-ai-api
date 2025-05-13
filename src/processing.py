import pkgutil

import yaml
from pydantic import BaseModel

from . import genai
from .entities import Document

PROMPTS = yaml.safe_load(pkgutil.get_data(__name__, "prompts.yaml"))


# Extract entities for the query and return the extract entities as an array
def extract_entities(user_query: str) -> list[str]:

    class ResponseFormat(BaseModel):
        entities: list[str]

    response: ResponseFormat = genai.generate_response(
        prompt=user_query,
        system_message=PROMPTS["extract_entities"],
        response_format=ResponseFormat,
    )
    return response.entities


def get_answer(user_question: str, documents: list[Document]) -> str:
    response = genai.generate_response(
        prompt=user_question,
        system_message=PROMPTS["answer_question"].format(documents=documents),
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
