"""
Load testing suite for the API.
"""

import os
from random import choice

from dotenv import load_dotenv
from locust import HttpUser, between, task

load_dotenv()


class WebsiteUser(HttpUser):
    wait_time = between(2, 5)

    def on_start(self):
        """
        Set properties when a user starts running.
        """
        # add the authentication header to be used for all requests in this HttpUser
        self.client.headers = {"X-Api-Key": os.environ["API_KEY"]}

    @task(weight=1)
    def open_documentation(self):
        """
        Visit documentation pages.
        """
        for path in ("/", "/docs", "/redoc", "/changelog"):
            self.client.get(path, name="documentation")

    @task(weight=2)
    def list_nodes(self):
        """
        List all the nodes in the graph
        """
        self.client.get("/nodes", name="/nodes")

    @task(weight=2)
    def get_node(self):
        """
        Get a specific node from the graph.
        """
        node_name = choice(
            [
                "COP28 Targets",
                "biochar",
                "electric mobility",
                "gas turbine",
                "hybrid energy storage systems",
                "liquefied natural gas",
                "nationally determined contributions",
                "ocean thermal energy conversion",
                "project gender markers",
                "public-private partnerships",
            ]
        )
        self.client.get(f"/nodes/{node_name}", name="/nodes")

    @task(weight=10)
    def query_graph(self):
        """
        Query the knowledge graph.
        """
        for query in ("climate change mitigation", "solar panels", "energy poverty"):
            self.client.get("/graph", params={"query": query}, name="/graph")

    @task(weight=10)
    def explore_graph(self):
        """
        Iteratively explore the knowledge graph.
        """
        query = "climate change mitigation"
        for _ in range(10):
            with self.client.get(
                "/graph", params={"query": query}, name="/graph", catch_response=True
            ) as response:
                # pick a new random node as a query
                data = response.json()
                node = choice(data["nodes"])
                query = node["name"]

    @task(weight=10)
    def ask_model(self):
        """
        Ask a natural language question to a model.
        """
        for message in (
            "How does climate change adaptation differ from climate change mitigation?",
            "How much energy does a typical residential solar panel generate?",
            "When was the Paris Agreement signed?",
        ):
            self.client.post(
                "/model", json=[{"role": "human", "content": message}], name="/model"
            )
