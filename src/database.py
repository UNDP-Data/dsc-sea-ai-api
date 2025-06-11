"""
Routines for database operations for RAG.
"""

import os

import lancedb

from . import genai, utils
from .entities import Document, Graph


async def get_connection() -> lancedb.AsyncConnection:
    """
    Get an asynchronous database connection for LanceDB stored on Azure Blob Storage.

    Returns
    -------
    lancedb.AsyncConnection
        Asynchronous database connection client.
    """
    return await lancedb.connect_async(
        "az://lancedb",
        storage_options={
            "account_name": os.environ["STORAGE_ACCOUNT_NAME"],
            "account_key": os.environ["STORAGE_ACCOUNT_KEY"],
        },
    )


class Client:
    """
    Database client class to perform routine operations using LanceDB.
    """

    def __init__(self, connection: lancedb.AsyncConnection):
        self.connection = connection

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
        vector = genai.embed_text(query)
        # perform a search to find the best match, i.e., a central node
        if not (
            results := await table_nodes.vector_search(vector)
            .limit(1)
            .select(["name"])
            .to_list()
        ):
            return Graph(nodes=[], edges=[])
        central_node_name = results[0]["name"]
        # extract a graph in a k-hop neighbourhood around the central node
        neighbourhoods, edges = {}, []
        subjects = (central_node_name,)
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
            await table_nodes.vector_search(vector)
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
            source=central_node_name,
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
        vector = genai.embed_text(query)
        return [
            Document(**doc)
            for doc in await table.vector_search(vector).limit(limit).to_list()
        ]
