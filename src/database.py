"""
Routines for database operations for RAG.
"""

import json
import os

import lancedb
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
    return {
        "account_name": os.environ["STORAGE_ACCOUNT_NAME"],
        "account_key": os.environ["STORAGE_ACCOUNT_KEY"],
    }


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
        table = await self.connection.open_table("nodes")
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
        table = await self.connection.open_table("nodes")
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

    async def find_graph(self, query: str, hops: int = 2) -> Graph:
        """
        Find a graph relevant to a given query.

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
        # open connections to node and edge tables
        table_nodes = await self.connection.open_table("nodes")
        table_edges = await self.connection.open_table("edges")
        # perform a search to find the best match, i.e., a central node
        central_node = await self.find_node(query, SearchMethod.VECTOR)
        # extract a graph in a k-hop neighbourhood around the central node
        neighbourhoods, edges = {}, []
        subjects = (central_node.name,)
        for hop in range(hops):
            # save the subjects as neighbourhood nodes
            neighbourhoods[hop] = subjects
            # get the outgoing edges for the subject(s)
            if not (
                edges_out := (
                    await table_edges.query()
                    .where(
                        f"subject in {subjects}"
                        if len(subjects) > 1
                        else f"subject == '{subjects[0]}'"
                    )
                    .to_list()
                )
            ):
                # exit if there are no more nodes in the neighbourhood
                break
            # save the edges
            edges.extend(edges_out)
            # extract and store adjacent nodes as new subjects
            subjects = tuple(edge["object"] for edge in edges_out)
        # save the outmost subjects after the last iteration too
        neighbourhoods[hops] = subjects
        # deduplicate the edges
        edges = {frozenset(edge.items()) for edge in edges}
        edges = list(map(dict, edges))
        # get all nodes for the edges in question
        node_names = tuple(
            {edge["subject"] for edge in edges} | {edge["object"] for edge in edges}
        )
        nodes = (
            await table_nodes.vector_search(central_node.vector)
            .distance_type("cosine")
            .where(
                (
                    f"name in {node_names}"
                    if node_names
                    else f"name == '{subjects[0]}'"
                ),  # handle an edge case when hops is set to zero
            )
            .limit(
                len(node_names) or 1  # handle an edge case when hops is set to zero too
            )  # add an explicit limit as per https://github.com/lancedb/lancedb/issues/1852
            .select(
                {
                    "name": "name",
                    "description": "description",
                    "weight": "1:float - _distance",  # SQL expression
                }
            )
            .to_list()
        )
        # construct a graph to easily traverse it
        metadata = utils.get_node_metadata(
            nodes=[node["name"] for node in nodes],
            edges=[(edge["subject"], edge["object"]) for edge in edges],
            source=central_node.name,
        )
        nodes = [node | metadata[node["name"]] for node in nodes]
        return Graph(nodes=nodes, edges=edges)

    async def retrieve_chunks(self, query: str, limit: int = 20) -> list[Chunk]:
        """
        Retrieve the document chunks from the database that best match a query.

        Parameters
        ----------
        query : str
            Plain text user query.
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
            .rerank(reranker, query_string=query)
            .limit(limit)
            .to_list()
        ]


# since the model uses the docstring, don't mention the artifacts there
@tool(parse_docstring=True, response_format="content_and_artifact")
async def retrieve_chunks(query: str) -> tuple[str, list[Document]]:
    """Retrieve relevant document chunks from the Sustainable Energy Academy database.

    The database can be used to answer questions on energy, climate change and
    sustainable development in general. Use the database to provide accurate and
    grounded responses. Make sure to cite sources in-line and provide a list of documents
    used at the end.

    Args:
        query (str): Plain text user query.

    Returns:
        str: JSON object containing the most relevant document chunks.
    """
    connection = await get_connection()
    client = Client(connection)
    chunks = await client.retrieve_chunks(query)
    data = json.dumps([chunk.to_context() for chunk in chunks])
    # deduplicate and sort
    documents = sorted(set(Document(**chunk.model_dump()) for chunk in chunks))
    return data, documents
