"""
Basic tests for `/graph` endpoint.
"""

import pytest

from tests.utils import validate_edge, validate_graph, validate_node


def test_graph_structure(test_client):
    """
    Test if the response from `/graph` endpoint has the expected format.
    """
    response = test_client.get("/graph", params={"query": "climate change mitigation"})
    assert response.status_code == 200
    data = response.json()
    validate_graph(data)
    list(map(validate_node, data["nodes"]))
    list(map(validate_edge, data["edges"]))


@pytest.mark.parametrize(
    "query,node_name",
    [
        ("climate change mitigation", "climate change scenario analysis"),
        ("solar energy", "solar energy"),
        ("energia solare", "solar energy"),  # Italian query
        ("Ηλιακή ενέργεια", "solar energy"),  # Greek query
        ("decarbonisation", "decarbonization"),
    ],
)
def test_graph_query(test_client, query: str, node_name: str):
    """
    Test if `/graph` endpoint produces expected response for various queries.
    """
    response = test_client.get("/graph", params={"query": query})
    assert response.status_code == 200
    data = response.json()
    nodes = data["nodes"]
    assert nodes[0]["neighbourhood"] == 0
    assert nodes[0]["name"] == node_name
    assert len({node["neighbourhood"] for node in nodes}) > 1


@pytest.mark.parametrize("hops", [0, 1, 2, 3])
def test_graph_hops(test_client, hops: int):
    """
    Test if `/graph` endpoint produces expected response for various settings of `hops`.
    """
    response = test_client.get("/graph", params={"query": "solar energy", "hops": hops})
    assert response.status_code == 200
    data = response.json()
    nodes, edges = data["nodes"], data["edges"]
    if hops == 0:
        assert len(nodes) == 1
        assert len(edges) == 0
    else:
        assert len(nodes) > 1
        assert len(edges) > 1
    assert all(node["neighbourhood"] <= hops for node in nodes)
