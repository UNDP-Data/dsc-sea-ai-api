"""
Routines for database operations for RAG.
"""

import os

import lancedb

from . import genai
from .entities import Document, Graph


def get_connection() -> lancedb.DBConnection:
    """
    Get a database connection for LanceDB stored on Azure Blob Storage.

    Returns
    -------
    lancedb.DBConnection
        Database connection client.
    """
    return lancedb.connect(
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

    def __init__(self):
        self.connection = get_connection()

    def extract_positions(self, neighbourhoods: dict[int, set[str]]) -> dict[str, int]:
        positions = {}
        # traverse from the closest neighbours
        for hop, node_names in sorted(neighbourhoods.items(), key=lambda x: x[0]):
            for node_name in node_names:
                # keep only the position closest to the central node
                if node_name not in positions:
                    positions[node_name] = hop
        return positions

    def find_graph(self, query: str, hops: int = 2) -> Graph:
        # open connections to node and edge tables
        table_nodes = self.connection.open_table("nodes")
        table_edges = self.connection.open_table("edges")
        # perform a full-text search to find the best match, i.e., a central node
        if not (
            results := table_nodes.search(query, query_type="fts")
            .limit(1)
            .select(["name"])
            .to_list()
        ):
            return Graph(nodes=[], edges=[])
        # extract a graph in a k-hop neighbourhood around the central node
        neighbourhoods, edges = {}, []
        subjects = (results[0]["name"],)  # start with the central node
        for hop in range(hops):
            # save the subjects as neighbourhood nodes
            neighbourhoods[hop] = subjects
            # get the outgoing edges for the subject(s)
            if not (
                edges_out := (
                    table_edges.search(None)
                    .where(
                        f"subject in {subjects}"
                        if len(subjects) > 1
                        else f'subject == "{subjects[0]}"'
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
        # extract the nodes and assign neighbourhood positions
        positions = self.extract_positions(neighbourhoods)
        nodes = table_nodes.search(None).where(f"name in {node_names}").to_list()
        nodes = [node | {"neighbourhood": positions[node["name"]]} for node in nodes]
        return Graph(nodes=nodes, edges=edges)

    def retrieve_documents(self, query: str, limit: int = 5) -> list[Document]:
        table = self.connection.open_table("documents")
        # perform a full-text search to find best matches
        vector = genai.embed_text(query)
        return table.search(vector).limit(limit).to_pydantic(Document)
