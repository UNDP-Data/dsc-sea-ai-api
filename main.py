"""
Entry point to the API.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Annotated

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src import database, genai
from src.entities import AssistantResponse, Graph, GraphParameters, Message

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
    connection = await database.get_connection()
    states = {"client": database.Client(connection)}
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
    response_model=Graph,
    response_model_by_alias=False,
)
async def query_knowledge_graph(
    request: Request,
    params: Annotated[GraphParameters, Query()],
):
    """
    Get a knowledge graph that best matches the query concept.
    """
    client: database.Client = request.state.client
    return await client.find_graph(**params.model_dump())


@app.post(
    path="/model",
    response_model=AssistantResponse,
    response_model_by_alias=False,
)
async def ask_model(request: Request, messages: list[Message]):
    """
    Ask a GenAI model to compose a response supplemented by knowledge graph data.

    Messages are expected to be in chronological order, i.e., from the oldest to most recent,
    with the current user message being the last one.
    """
    if messages[-1].role != "human":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The last message must come from the user.",
        )
    user_query = messages[-1].content
    client: database.Client = request.state.client
    documents, entities, ideas = await asyncio.gather(
        client.retrieve_documents(user_query),
        genai.extract_entities(user_query),
        genai.generate_query_ideas(user_query),
    )
    graphs = await asyncio.gather(*[client.find_graph(entity) for entity in entities])
    response = {
        "role": "assistant",
        "content": await genai.get_answer(user_query, documents, messages),
        "ideas": ideas or None,
        "documents": documents,
        "graph": sum(graphs, Graph(nodes=[], edges=[])),  # merge all graphs
    }
    return response
