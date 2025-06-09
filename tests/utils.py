"""
Utility functions for validating data structures in responses.
"""

__all__ = ["validate_graph", "validate_node", "validate_edge"]


def validate_graph(graph):
    """
    Validate basic graph data structure.
    """
    assert isinstance(graph, dict)
    nodes = graph.get("nodes")
    assert isinstance(nodes, list)
    assert len(nodes) > 1
    edges = graph.get("edges")
    assert isinstance(edges, list)
    assert len(edges) > 1


def validate_node(node):
    """
    Validate basic node data structure.
    """
    assert isinstance(node, dict)
    assert isinstance(node.get("neighbourhood"), int)
    assert isinstance(node.get("name"), str)
    assert isinstance(node.get("weight"), float)
    assert 0.0 <= node["weight"] <= 1.0
    assert isinstance(node.get("colour"), str)
    assert node["colour"].startswith("#")


def validate_edge(edge):
    """
    Validate basic edge data structure.
    """
    assert isinstance(edge, dict)
    assert isinstance(edge.get("subject"), str)
    assert isinstance(edge.get("object"), str)
    assert edge["subject"] != edge["object"]
