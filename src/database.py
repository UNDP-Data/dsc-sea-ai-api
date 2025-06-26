"""
Routines for database operations for RAG.
"""

import json
import os

import lancedb
from langchain_core.tools import tool

from . import genai, utils
from .entities import Document, Graph, Node, SearchMethod

__all__ = ["STORAGE_OPTIONS", "get_connection", "Client", "retrieve_documents"]

STORAGE_OPTIONS = {
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
    return await lancedb.connect_async("az://lancedb", storage_options=STORAGE_OPTIONS)


class Client:
    """
    Database client class to perform routine operations using LanceDB.
    """

    def __init__(self, connection: lancedb.AsyncConnection):
        self.connection = connection
        self.embedder = genai.get_embedding_client()

    async def list_nodes(self) -> list[Node]:
        """
        List all nodes in the graph.

        Returns
        -------
        list[Node]
            A list of nodes sorted by name.
        """
        table = await self.connection.open_table("nodes")
        return sorted([Node(**node) for node in await table.query().to_list()])

    async def find_node(
        self, query: str, method: SearchMethod, with_vector: bool = True
    ) -> Node | None:
        """
        Find the node that best matches the query.

        Parameters
        ----------
        query : str
            Plain text query.
        method : {SearchMethod.EXACT, SearchMethod.VECTOR}
            Search method to utilise.
        with_vector : bool, default=True
            If True, include the node's vector in `metadata` property.

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
        node = nodes[0]
        metadata = node.pop("metadata")
        if with_vector:
            metadata |= {"vector": node.pop("vector")}
        return Node(**node, metadata=metadata)

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
            await table_nodes.vector_search(central_node.metadata["vector"])
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
                    "metadata": "metadata",
                    "weight": "1 - _distance",  # SQL expression
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

    async def retrieve_documents(self, query: str, limit: int = 5) -> list[Document]:
        """
        Retrieve the documents from the database that best match a query.

        Parameters
        ----------
        query : str
            Plain text user query.
        limit : int, default=5
            Maximum number of best matching documents to retrieve.

        Returns
        -------
        list[Document]
            List of most relevant documents.
        """
        table = await self.connection.open_table("documents")
        # perform a vector search to find best matches
        vector = await self.embedder.aembed_query(query)
        return [
            Document(**doc)
            for doc in await table.vector_search(vector).limit(limit).to_list()
        ]


@tool(parse_docstring=True)
async def retrieve_documents(query: str) -> str:
    """Retrieve relevant documents from the Sustainable Energy Academy database.

    The database can be used to answer questions to energy, climate change and
    sustainable development in general. Use the database to provide accurate and
    grounded responses.

    Args:
        query (str): Plain text user query.

    Returns:
        str: JSON object containing the most relevant document chunks.
    """
    connection = await get_connection()
    client = Client(connection)
    documents = await client.retrieve_documents(query)
    data = json.dumps([document.model_dump() for document in documents])
    return data
