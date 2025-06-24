"""
Entry point to the API.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Annotated

import yaml
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
    path="/nodes",
    response_model=list[Node],
    response_model_by_alias=False,
    dependencies=[Depends(authenticate)],
)
async def list_nodes(request: Request):
    """
    List all nodes in the graph. Since there is no central node,
    `neighbourhood` is set to zero for all nodes.
    """
    client: database.Client = request.state.client
    return await client.list_nodes()


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
    if (
        node := await client.find_node(name, SearchMethod.EXACT, with_vector=False)
    ) is None:
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
    return await client.find_graph(**params.model_dump())


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
    entities, ideas = await asyncio.gather(
        genai.extract_entities(user_query),
        genai.generate_query_ideas(user_query),
    )
    graphs = await asyncio.gather(*[client.find_graph(entity) for entity in entities])
    response = AssistantResponse(
        role="assistant",
        content="",
        ideas=ideas or None,
        documents=None,
        graph=sum(graphs, Graph(nodes=[], edges=[])),  # merge all graphs
    )
    return StreamingResponse(
        content=genai.get_answer(
            messages, response, tools=[database.retrieve_documents]
        ),
        media_type="application/x-ndjson",
    )
