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

    def find_graph(self, query: str) -> Graph:
        # open connections to node and edge tables
        table_nodes = self.connection.open_table("nodes")
        table_edges = self.connection.open_table("edges")
        # perform a full-text search to find the best match
        if not (
            results := table_nodes.search(query, query_type="fts")
            .limit(1)
            .select(["name"])
            .to_list()
        ):
            return Graph(nodes=[], edges=[])
        # extract the best matching subject (concept)
        subject = results[0]["name"]
        # get the outgoing edges for the subject
        edges = table_edges.search(None).where(f'subject == "{subject}"').to_list()
        # get the outgoing edges for objects the subject is related to (1-hop neighbourhood)
        objects = tuple(edge["object"] for edge in edges)
        edges.extend(table_edges.search(None).where(f"subject in {objects}").to_list())
        # get all nodes for the edges in question
        entities = tuple(
            {edge["subject"] for edge in edges} | {edge["object"] for edge in edges}
        )
        nodes = table_nodes.search(None).where(f"name in {entities}").to_list()
        return Graph(nodes=nodes, edges=edges)

    def retrieve_documents(self, query: str, limit: int = 5) -> list[Document]:
        table = self.connection.open_table("documents")
        # perform a full-text search to find best matches
        vector = genai.embed_text(query)
        return table.search(vector).limit(limit).to_pydantic(Document)
