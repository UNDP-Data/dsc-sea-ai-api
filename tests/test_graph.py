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
    "query",
    [
        "climate change mitigation",
        "solar energy",
        "energia solare",  # Italian query
        "Ηλιακή ενέργεια",  # Greek query
        "decarbonisation",
    ],
)
def test_graph_query(test_client, query: str):
    """
    Test if `/graph` endpoint produces expected response for various queries.
    """
    response = test_client.get("/graph", params={"query": query})
    assert response.status_code == 200
    data = response.json()
    nodes = data["nodes"]
    assert any([node["neighbourhood"] == 0 for node in nodes])
    assert len(nodes) > 1
    assert any(isinstance(node.get("name"), str) and node["name"].strip() for node in nodes)


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
