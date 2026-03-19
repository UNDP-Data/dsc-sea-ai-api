"""
Basic tests for `/graph/v2` endpoint.
"""

from tests.utils import validate_edge, validate_graph


def test_graph_v2_structure(test_client):
    """
    Test if `/graph/v2` has expected top-level shape.
    """
    response = test_client.get("/graph/v2", params={"query": "climate change mitigation"})
    assert response.status_code == 200
    data = response.json()
    validate_graph(data)
    list(map(validate_edge, data["edges"]))


def test_graph_v2_tiers(test_client):
    """
    Test if `/graph/v2` provides valid node tiers and limits central nodes.
    """
    response = test_client.get("/graph/v2", params={"query": "solar energy transition policy"})
    assert response.status_code == 200
    data = response.json()
    nodes = data["nodes"]
    tiers = {node["tier"] for node in nodes}
    assert tiers <= {"central", "secondary", "periphery"}
    central_nodes = [node for node in nodes if node["tier"] == "central"]
    assert 1 <= len(central_nodes) <= 3
    assert all(isinstance(node.get("colour"), str) and node["colour"].startswith("#") for node in nodes)


def test_graph_v2_edges_reference_nodes(test_client):
    """
    Test if all edges reference nodes returned in the same response.
    """
    response = test_client.get("/graph/v2", params={"query": "decarbonization and grid resilience"})
    assert response.status_code == 200
    data = response.json()
    node_names = {node["name"] for node in data["nodes"]}
    assert all(edge["subject"] in node_names for edge in data["edges"])
    assert all(edge["object"] in node_names for edge in data["edges"])
    assert all("level" not in edge for edge in data["edges"])
