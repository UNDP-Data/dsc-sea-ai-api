"""
Entry point to the API.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import networkx as nx
import yaml
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from packaging.version import Version

from src import database, genai
from src.entities import (
    AssistantResponse,
    Graph,
    GraphParameters,
    Message,
    Node,
    SearchMethod,
)
from src.security import authenticate

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
    client = database.Client(connection)
    states = {"client": client, "graph": await client.get_knowledge_graph()}
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
    file_paths = sorted(
        Path("templates", "changelog").glob("*.html"),
        reverse=True,
        key=lambda path: Version(path.with_suffix("").name.lstrip("v")),
    )
    file_names = [file_path.name for file_path in file_paths]
    return templates.TemplateResponse(
        request=request,
        name="changelog.html",
        context={"file_names": file_names},
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    Use to display the favicon in the homepage. For favicon on the documentation pages,
    an override of /docs and /redoc endpoints is required.
    """
    return FileResponse("./static/favicon.ico")


@app.get(
    path="/nodes",
    response_model=list[Node],
    response_model_by_alias=False,
    dependencies=[Depends(authenticate)],
)
async def search_nodes(
    request: Request,
    pattern: Annotated[str | None, Query(max_length=50)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
):
    """
    Search nodes in the graph. Since there is no central node,
    `neighbourhood` is set to zero for all nodes.
    """
    client: database.Client = request.state.client
    return await client.search_nodes(pattern or "", limit)


@app.get(
    path="/nodes/{name}",
    response_model=Node,
    response_model_by_alias=False,
    dependencies=[Depends(authenticate)],
)
async def get_node(request: Request, name: str):
    """
    Get a single node by name. Case insensitive.
    """
    client: database.Client = request.state.client
    if (node := await client.find_node(name, SearchMethod.EXACT)) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node '{name}' does not exist.",
        )
    return node


@app.get(
    path="/graph",
    response_model=Graph,
    response_model_by_alias=False,
    dependencies=[Depends(authenticate)],
)
async def query_knowledge_graph(
    request: Request,
    params: Annotated[GraphParameters, Query()],
):
    """
    Get a knowledge graph that best matches the query concept.
    """
    client: database.Client = request.state.client
    graph: nx.Graph = request.state.graph
    return await client.find_subgraph(graph, **params.model_dump())


@app.post(
    path="/model",
    response_model=AssistantResponse,
    response_model_by_alias=False,
    dependencies=[Depends(authenticate)],
)
async def ask_model(request: Request, messages: list[Message]):
    """
    Ask a GenAI model to compose a response supplemented by knowledge graph data.

    Messages are expected to be in chronological order, i.e., from the oldest to most recent,
    with the current user message being the last one. This endpoint streams the response in
    NDJSON format.
    """
    if messages[-1].role != "human":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The last message must come from the user.",
        )
    user_query = messages[-1].content
    client: database.Client = request.state.client
    graph: nx.Graph = request.state.graph
    entities = await genai.extract_entities(user_query)
    graph = await client.find_subgraph(graph, entities)
    response = AssistantResponse(
        role="assistant",
        content="",
        graph=graph,
    )
    datasets = [await client.get_sdg7_dataset()]
    tools = [database.retrieve_chunks] + genai.get_sql_tools(datasets)
    return StreamingResponse(
        content=genai.get_answer(messages, response, tools=tools),
        media_type="application/x-ndjson",
    )
