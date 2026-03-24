"""
Entry point to the API.
"""

import asyncio
from contextlib import asynccontextmanager
from inspect import isawaitable
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4
import json
import logging

import networkx as nx
import yaml
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
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
from src.kg import v1 as kg_v1
from src.kg import v2 as kg_v2
from src.kg.types import GraphV2, GraphV2Parameters
from src.security import authenticate

load_dotenv()
logger = logging.getLogger(__name__)


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
    return await kg_v1.build_subgraph_v1(client, graph, **params.model_dump())


@app.get(
    path="/graph/v2",
    response_model=GraphV2,
    response_model_by_alias=False,
    dependencies=[Depends(authenticate)],
)
async def query_knowledge_graph_v2(
    request: Request,
    response: Response,
    params: Annotated[GraphV2Parameters, Query()],
):
    """
    Get a staged knowledge graph with central / secondary / periphery tiers.
    """
    client: database.Client = request.state.client
    graph: nx.Graph = request.state.graph
    timings: dict[str, float] = {}
    result = await kg_v2.build_subgraph_v2(client, graph, params.query, timings=timings)
    response.headers["X-KG-Timing"] = json.dumps(timings, separators=(",", ":"))
    server_timing = []
    key_to_metric = {
        "central_exact_match_ms": "central_exact",
        "central_lexical_ms": "central_lexical",
        "central_embed_ms": "central_embed",
        "central_vector_search_ms": "central_search",
        "central_selection_ms": "central",
        "stage1_ms": "stage1",
        "stage2_ms": "stage2",
        "stage3_ms": "stage3",
        "total_ms": "total",
    }
    for key, metric in key_to_metric.items():
        if key in timings:
            server_timing.append(f"{metric};dur={timings[key]:.2f}")
    if server_timing:
        response.headers["Server-Timing"] = ", ".join(server_timing)
    return result


@app.get(
    path="/debug/tables",
    dependencies=[Depends(authenticate)],
    include_in_schema=False,
)
async def debug_tables(request: Request):
    """
    Return table presence and row counts to validate storage wiring.
    """
    async def maybe_await(value):
        return await value if isawaitable(value) else value

    client: database.Client = request.state.client
    names = await maybe_await(client.connection.table_names())
    names = list(names)
    status = {}
    for name in ("nodes", "edges", "chunks", "sdg7"):
        if name in names:
            try:
                table = await maybe_await(client.connection.open_table(name))
                rows = await maybe_await(table.count_rows())
                status[name] = {"exists": True, "rows": rows}
            except Exception as error:
                status[name] = {"exists": True, "rows": None, "error": str(error)}
        else:
            status[name] = {"exists": False, "rows": 0}
    return {"tables": status, "all_tables": sorted(names)}


@app.post(
    path="/model",
    response_model=AssistantResponse,
    response_model_by_alias=False,
    dependencies=[Depends(authenticate)],
)
async def ask_model(
    request: Request,
    messages: list[Message],
    graph_version: Annotated[
        Literal["v1", "v2"],
        Query(
            description="Knowledge graph response schema version.",
            json_schema_extra={"example": "v2"},
        ),
    ] = "v2",
):
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
    request_id = request.headers.get("X-Request-Id") or uuid4().hex
    user_query = messages[-1].content
    client: database.Client = request.state.client
    graph: nx.Graph = request.state.graph
    logger.info(
        "Model request started request_id=%s graph_version=%s",
        request_id,
        graph_version,
    )

    async def get_response_graph() -> Graph | GraphV2:
        if graph_version == "v1":
            entities = await genai.extract_entities(user_query)
            return await client.find_subgraph(graph, entities)
        return await kg_v2.build_subgraph_v2(client, graph, user_query)

    async def stream_model_response():
        emitted_chunks = 0
        if await request.is_disconnected():
            logger.info(
                "Client disconnected before /model stream started request_id=%s.",
                request_id,
            )
            return
        queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue()

        async def produce_graph() -> None:
            try:
                response_graph = await get_response_graph()
            except Exception as error:
                logger.exception(
                    "Failed to build response graph request_id=%s: %s",
                    request_id,
                    error,
                )
                return
            if await request.is_disconnected():
                logger.info(
                    "Client disconnected before graph chunk was sent request_id=%s.",
                    request_id,
                )
                return
            graph_response = AssistantResponse(
                role="assistant",
                content="",
                graph=response_graph,
            )
            await queue.put(("data", graph_response.model_dump_json() + "\n"))

        async def produce_answer() -> None:
            tools = [database.retrieve_chunks]
            try:
                datasets = [await client.get_sdg7_dataset()]
                tools += genai.get_sql_tools(datasets)
            except Exception as error:
                logger.exception(
                    "Failed to prepare SQL tools request_id=%s: %s",
                    request_id,
                    error,
                )
            response = AssistantResponse(
                role="assistant",
                content="",
                graph=None,
            )
            try:
                async for chunk in genai.get_answer(messages, response, tools=tools):
                    if await request.is_disconnected():
                        logger.info(
                            "Client disconnected during /model token stream request_id=%s.",
                            request_id,
                        )
                        return
                    await queue.put(("data", chunk))
            except Exception as error:
                logger.exception(
                    "Failed while streaming answer request_id=%s: %s",
                    request_id,
                    error,
                )
                fallback = AssistantResponse(
                    role="assistant",
                    content="I ran into a temporary issue while generating a response. Please retry your question.",
                    graph=None,
                )
                await queue.put(("data", fallback.model_dump_json() + "\n"))

        async def run_producer(done_label: str, producer) -> None:
            try:
                await producer()
            finally:
                await queue.put((done_label, None))

        producer_tasks = [
            asyncio.create_task(run_producer("graph_done", produce_graph)),
            asyncio.create_task(run_producer("answer_done", produce_answer)),
        ]
        completed: set[str] = set()
        try:
            while len(completed) < 2:
                if await request.is_disconnected():
                    logger.info(
                        "Client disconnected before /model stream completion request_id=%s.",
                        request_id,
                    )
                    break
                try:
                    label, payload = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if payload is None:
                    completed.add(label)
                    continue
                yield payload
                emitted_chunks += 1
        finally:
            for task in producer_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*producer_tasks, return_exceptions=True)
        logger.info(
            "Model request finished request_id=%s chunks=%s",
            request_id,
            emitted_chunks,
        )

    return StreamingResponse(
        content=stream_model_response(),
        media_type="application/x-ndjson",
        headers={"X-Request-Id": request_id},
    )
