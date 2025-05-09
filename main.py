"""
Entry point to the API.
"""

from contextlib import asynccontextmanager
from typing import Annotated

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request

from src import database, processing
from src.entities import AssistantMessage, HumanMessage, KnowledgeGraph

load_dotenv()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Define the logic that should be executed before the application start up.

    See the details in [the documentation](https://fastapi.tiangolo.com/advanced/events/).

    Parameters
    ----------
    _ : FastAPI
        Application object, not directly used but required by signature.

    Yields
    ------
    states : dict
        Dictionary of arbitrary state variables.
    """
    states = {"client": database.Client.from_model()}
    yield states


with open("metadata.yaml", "r", encoding="utf-8") as file:
    metadata = yaml.safe_load(file)
app = FastAPI(**metadata, lifespan=lifespan)


@app.get(
    path="/kg_query",
    response_model=KnowledgeGraph,
    response_model_by_alias=False,
)
async def get_kg_data(q: Annotated[list[str], Query()]):
    subgraphs = processing.find_kg(q)
    return {"subgraphs": subgraphs}


@app.post(
    path="/llm",
    response_model=AssistantMessage,
    response_model_by_alias=False,
)
async def send_prompt_llm(request: Request, message: HumanMessage):
    user_query = message.content
    response = {}
    client: database.Client = request.state.client
    # user is requering ... get all relevant answers
    entities_dict = processing.get_knowledge_graph(user_query)
    query_idea_list = processing.generate_query_ideas(user_query) or None
    entities_array = entities_dict["entities"] or None
    if message.full:
        excerpts_dict = client.process_queries(user_query)
        excerpts_dict_synthesis = processing.remove_thumbnails(excerpts_dict)
        answer = processing.get_answer(user_query, excerpts_dict_synthesis)
        subgraphs = None
    else:
        excerpts_dict, answer = {}, "Processing final answer... "
        subgraphs = processing.find_kg(entities_array)

    response = {
        "content": answer,
        "entities": entities_array,
        "query_ideas": query_idea_list,
        "excerpts": excerpts_dict,
        "subgraphs": subgraphs,
    }

    # Return the response
    return response
