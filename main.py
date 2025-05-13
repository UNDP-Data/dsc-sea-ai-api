"""
Entry point to the API.
"""

from contextlib import asynccontextmanager
from typing import Annotated

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates/")


@app.get(
    path="/",
    include_in_schema=False,
)
async def root(request: Request):
    """
    Return the homepage template.
    """
    return templates.TemplateResponse(request=request, name="index.html")


@app.get(
    path="/changelog",
    include_in_schema=False,
)
async def changelog(request: Request):
    """
    Return the changelog template.
    """
    return templates.TemplateResponse(request=request, name="changelog.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    Use to display the favicon in the homepage. For favicon on the documentation pages,
    an override of /docs and /redoc endpoints is required.
    """
    return FileResponse("./static/favicon.ico")


@app.get(
    path="/graph",
    response_model=KnowledgeGraph,
    response_model_by_alias=False,
)
async def query_knowledge_graph(
    query: Annotated[
        list[str],
        Query(
            description="One or more queries to retrieve relevant concepts for",
            example=["climate change mitigation"],
        ),
    ],
):
    """
    Get relevant subgraphs for query concepts from the knowledge graph.
    """
    subgraphs = processing.find_kg(query)
    return {"subgraphs": subgraphs}


@app.post(
    path="/model",
    response_model=AssistantMessage,
    response_model_by_alias=False,
)
async def ask_model(request: Request, message: HumanMessage):
    """
    Ask a GenAI model to compose a response supplemented by knowledge graph data.
    """
    user_query = message.content
    response = {}
    client: database.Client = request.state.client
    # user is requering ... get all relevant answers
    entity_list = processing.extract_entities(user_query)
    query_idea_list = processing.generate_query_ideas(user_query) or None
    entities_array = entity_list or None
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
