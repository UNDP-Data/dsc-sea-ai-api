"""
Routines for database operations for RAG.
"""

import asyncio
import json
import os
from urllib.parse import urlparse

import lancedb
import networkx as nx
import pandas as pd
import pyarrow as pa
from lancedb.rerankers import Reranker
from langchain_core.tools import tool

from . import genai, utils
from .entities import Chunk, Document, Graph, Node, SearchMethod

__all__ = ["get_storage_options", "get_connection", "Client", "retrieve_chunks"]


def get_storage_options() -> dict[str, str]:
    """
    Get storage options for Azure Blob Storage backend.

    The options can be passed to LanceDB or pandas to connect to remote storage.

    Returns
    -------
    dict
        Mapping storage options.
    """
    sas_url = os.getenv("STORAGE_SAS_URL")
    if sas_url:
        parsed = urlparse(sas_url)
        host = parsed.netloc.split(":")[0]
        account_name = host.split(".")[0] if host else ""
        query = parsed.query.lstrip("?")
        if account_name and query:
            return {"account_name": account_name, "sas_token": query}
        raise RuntimeError(
            "Invalid `STORAGE_SAS_URL`. Expected a full Azure Blob SAS URL like "
            "`https://<account>.blob.core.windows.net/...?...`."
        )

    account_name = os.getenv("STORAGE_ACCOUNT_NAME")
    account_key = os.getenv("STORAGE_ACCOUNT_KEY")
    sas_token = os.getenv("STORAGE_SAS_TOKEN")
    if not account_name:
        raise RuntimeError(
            "Missing `STORAGE_ACCOUNT_NAME` environment variable. "
            "Populate it in your `.env` file."
        )
    if account_key:
        return {"account_name": account_name, "account_key": account_key}
    if sas_token:
        return {"account_name": account_name, "sas_token": sas_token}
    raise RuntimeError(
        "Missing storage credentials. Set either `STORAGE_ACCOUNT_KEY` or "
        "`STORAGE_SAS_TOKEN` in your `.env` file."
    )


async def get_connection() -> lancedb.AsyncConnection:
    """
    Get an asynchronous database connection for LanceDB stored on Azure Blob Storage.

    Returns
    -------
    lancedb.AsyncConnection
        Asynchronous database connection client.
    """
    return await lancedb.connect_async(
        "az://lancedb", storage_options=get_storage_options()
    )


class TimeReranker(Reranker):
    """
    Custom reranker class to prioritise more recent documents.

    See https://lancedb.github.io/lancedb/reranking/custom_reranker.
    """

    def rerank_hybrid(
        self, query: str, vector_results: pa.Table, fts_results: pa.Table
    ):
        """
        Required method for reranking.
        """
        return self.merge_results(vector_results, fts_results)

    def rerank_vector(self, query: str, vector_results: pa.Table):
        """
        Rerank vector search results to promote more recent documents.
        """
        df = vector_results.to_pandas()
        # use exponential decay function to increase the distance for older documents
        df["_distance"] = df.eval("_distance / exp(-0.1 * (2025 - year))")
        df.sort_values("_distance", ignore_index=True, inplace=True)
        return pa.Table.from_pandas(df)


class Client:
    """
    Database client class to perform routine operations using LanceDB.
    """

    def __init__(self, connection: lancedb.AsyncConnection):
        self.connection = connection
        self.embedder = genai.get_embedding_client()

    async def search_nodes(self, pattern: str = "", limit: int = 10) -> list[Node]:
        """
        Search nodes in the graph, optionally utilising RegEx patterns.

        The search is case insentitive.

        Parameters
        ----------
        pattern : str, optional
            Optional pattern to match the nodes.
        limit : int, default=10
            Number of matching nodes to return.

        Returns
        -------
        list[Node]
            A list of nodes sorted by name.
        """
        try:
            table = await self.connection.open_table("nodes")
        except ValueError as error:
            if "was not found" in str(error):
                return []
            raise
        return sorted(
            [
                Node(**node)
                for node in await table.query()
                .where(f"regexp_match(name, '(?i){pattern}')")
                .limit(limit)
                .to_list()
            ]
        )

    async def find_node(self, query: str, method: SearchMethod) -> Node | None:
        """
        Find the node that best matches the query.

        Parameters
        ----------
        query : str
            Plain text query.
        method : {SearchMethod.EXACT, SearchMethod.VECTOR}
            Search method to utilise.

        Returns
        -------
        Node | None
            A matching node if found, otherwise None is returned.
        """
        try:
            table = await self.connection.open_table("nodes")
        except ValueError as error:
            if "was not found" in str(error):
                return None
            raise
        match method:
            case SearchMethod.VECTOR:
                vector = await self.embedder.aembed_query(query)
                results = table.vector_search(vector)
            case SearchMethod.EXACT:
                # case insensitive
                results = table.query().where(f"lower(name) == '{query.lower()}'")
            case _:
                raise ValueError(f"Method {method} is not supported.")
        if not (nodes := await results.limit(1).to_list()):
            return None
        return Node(**nodes[0])

    async def find_subgraph(
        self, graph: nx.Graph, query: str | list[str], hops: int = 2
    ) -> Graph:
        """
        Find a subgraph relevant to a given query.

        Parameters
        ----------
        query : str
            Plain text user query.
        hops : int, default=2
            The size of the neighbourhood to extract the graph from.

        Returns
        -------
        Graph
            Object with node and edge lists.
        """
        # match query or queries to find central nodes
        queries = [query] if isinstance(query, str) else query
        central_nodes = await asyncio.gather(
            *[self.find_node(query, SearchMethod.VECTOR) for query in queries]
        )
        # subset the graph using central nodes as sources
        sources = [node.name for node in central_nodes if node is not None]
        if not sources:
            return Graph(nodes=[], edges=[])
        nodes = utils.get_neighbourhood_nodes(graph, sources, hops)
        graph = graph.subgraph(nodes).copy()
        nodes = utils.get_closest_nodes(graph, sources)
        graph = graph.subgraph(nodes).copy()
        graph = utils.prune_edges(graph)
        # pass all sources to compute neighbourhood from nearest central node
        return Graph.from_networkx(graph, sources)

    async def retrieve_chunks(
        self, query: str, where: str = "", limit: int = 20
    ) -> list[Chunk]:
        """
        Retrieve the document chunks from the database that best match a query.

        Parameters
        ----------
        query : str
            Plain text user query.
        where : str, optional
            Filtering conditions for the where clause.
        limit : int, default=20
            Maximum number of best matching chunks to retrieve.

        Returns
        -------
        list[Chunk]
            List of most relevant document chunks.
        """
        table = await self.connection.open_table("chunks")
        # perform a vector search to find best matches
        vector = await self.embedder.aembed_query(query)
        reranker = TimeReranker()
        return [
            Chunk(**chunk)
            for chunk in await table.vector_search(vector)
            .where(where)
            .rerank(reranker, query_string=query)
            .limit(limit)
            .to_list()
        ]

    async def get_sdg7_dataset(self) -> pd.DataFrame:
        """
        Get SDG 7 dataset.

        Returns
        -------
        pd.DataFrame
            Pandas data frame with SDG 7 indicators.
        """
        table = await self.connection.open_table("sdg7")
        df = await table.to_pandas()
        df.name = "indicators"
        return df

    async def get_knowledge_graph(self) -> nx.Graph:
        """
        Create a (weighted directed) knowledge graph from database tables.

        Returns
        -------
        nx.Graph
            Knowledge graph.
        """
        try:
            table = await self.connection.open_table("nodes")
        except ValueError as error:
            if "was not found" in str(error):
                return nx.DiGraph()
            raise
        df = await table.query().select(["name", "description", "weight"]).to_pandas()
        nodes = zip(
            df["name"].tolist(),
            df[["description", "weight"]].to_dict(orient="records"),
        )
        try:
            table = await self.connection.open_table("edges")
        except ValueError as error:
            if "was not found" in str(error):
                graph = nx.DiGraph()
                graph.add_nodes_from(nodes)
                return graph
            raise
        # check if level column exists in the table schema
        schema = await table.schema()
        edge_columns = ["subject", "object", "predicate", "description", "weight"]
        edge_attrs = ["predicate", "description", "weight"]
        if "level" in [field.name for field in schema]:
            edge_columns.append("level")
            edge_attrs.append("level")
        df = await table.query().select(edge_columns).to_pandas()
        edges = zip(
            df["subject"].tolist(),
            df["object"].tolist(),
            df[edge_attrs].to_dict(orient="records"),
        )
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph


# since the model uses the docstring, don't mention the artifacts there
@tool(parse_docstring=True, response_format="content_and_artifact")
async def retrieve_chunks(
    query: str, years: int | list[int] = 2025
) -> tuple[str, list[Document]]:
    """Retrieve relevant document chunks from the Sustainable Energy Academy database.

    The database can be used to answer questions on energy, climate change and
    sustainable development in general. Use the database to provide accurate and
    grounded responses. Make sure to cite sources in-line and provide a list of documents
    used at the end.

    Args:
        query (str): Plain text user query.
        years (Union[int, Tuple[int, ...]]): Specific year or years the user refers to if
            applicable. Otherwise, use the current year.

    Returns:
        str: JSON object containing the most relevant document chunks.
    """
    connection = await get_connection()
    client = Client(connection)
    where = (
        f"year = {years}"
        if isinstance(years, int)
        else f"year IN ({', '.join(map(str, years))})"
    )
    chunks = await client.retrieve_chunks(query, where)
    data = json.dumps([chunk.to_context() for chunk in chunks])
    # deduplicate and sort
    documents = sorted(set(Document(**chunk.model_dump()) for chunk in chunks))
    return data, documents
