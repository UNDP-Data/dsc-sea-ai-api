"""
Basic tests for `/nodes` endpoints.
"""

import pytest


def test_list_nodes(test_client):
    """
    Test if `/nodes` endpoint produces the expected response.
    """
    response = test_client.get("/nodes")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 10
    assert all(isinstance(node, dict) for node in data)
    # no central node
    assert all(node.get("neighbourhood") == 0 for node in data)
    assert any(
        node.get("name", "").lower() == "climate adaptation strategies" for node in data
    )


@pytest.mark.parametrize(
    "name",
    ["climate adaptation strategies", "Climate adAPTation STRATEGIES", "solar energy"],
)
def test_get_node_200(test_client, name: str):
    """
    Test if `/nodes/{name}` endpoint correctly finds nodes.
    """
    response = test_client.get(f"/nodes/{name}")
    assert response.status_code == 200
    node = response.json()
    assert isinstance(node, dict)
    assert node["neighbourhood"] == 0
    assert node["name"].upper() == name.upper()


@pytest.mark.parametrize(
    "name",
    [" solar energy ", "energia solare", "decarbonisation"],
)
def test_get_node_404(test_client, name: str):
    """
    Test if `/nodes/{name}` endpoint correctly raises 404 if there is no matching node.
    """
    response = test_client.get(f"/nodes/{name}")
    assert response.status_code == 404
