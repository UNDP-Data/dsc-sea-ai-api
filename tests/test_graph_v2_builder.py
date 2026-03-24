"""
Unit tests for staged V2 builder logic on synthetic graphs.
"""

import asyncio

import networkx as nx

from src import graph_v2
from src.graph_v2 import BranchingConfig


class _FakeVectorSearch:
    def __init__(self, rows):
        self._rows = rows

    def limit(self, _n):
        return self

    def select(self, _columns):
        return self

    async def to_list(self):
        return self._rows


class _FakeTable:
    def __init__(self, rows):
        self._rows = rows

    def vector_search(self, _vector):
        return _FakeVectorSearch(self._rows)


class _FakeConnection:
    def __init__(self, rows):
        self._rows = rows

    async def open_table(self, _name):
        return _FakeTable(self._rows)


class _FakeEmbedder:
    async def aembed_query(self, _query):
        return [0.1, 0.2, 0.3]


class _FakeClient:
    def __init__(self, rows):
        self.connection = _FakeConnection(rows)
        self.embedder = _FakeEmbedder()


def _make_synthetic_graph() -> nx.DiGraph:
    graph = nx.DiGraph()

    def add_node(name, weight=1.0):
        graph.add_node(name, description=f"{name} description", weight=float(weight))

    centres = ["c1", "c2"]
    for centre in centres:
        add_node(centre, weight=4.0)

    first_nodes = []
    for centre_index, centre in enumerate(centres, start=1):
        for index in range(1, 8):
            node = f"s{centre_index}_{index}"
            first_nodes.append(node)
            add_node(node, weight=3.0 - index * 0.1)
            graph.add_edge(
                centre,
                node,
                predicate=f"domain_{index % 3}",
                description=f"{centre} to {node}",
                weight=4.0 - index * 0.2,
            )

    for parent in first_nodes:
        for index in range(1, 6):
            child = f"x_{parent}_{index}"
            add_node(child, weight=2.0 - index * 0.1)
            graph.add_edge(
                parent,
                child,
                predicate=f"branch_{index % 2}",
                description=f"{parent} to {child}",
                weight=2.8 - index * 0.2,
            )
            for leaf_index in range(1, 4):
                leaf = f"p_{child}_{leaf_index}"
                add_node(leaf, weight=1.0)
                graph.add_edge(
                    child,
                    leaf,
                    predicate="leaf",
                    description=f"{child} to {leaf}",
                    weight=1.8 - leaf_index * 0.1,
                )

    return graph


def test_build_subgraph_v2_tier_and_level_limits(monkeypatch):
    graph = _make_synthetic_graph()

    async def fake_find_central_nodes(client, graph_arg, query):
        return ["c1", "c2"]

    monkeypatch.setattr(graph_v2, "_find_central_nodes", fake_find_central_nodes)
    result = asyncio.run(
        graph_v2.build_subgraph_v2(client=object(), graph=graph, query="grid policy")
    )

    tiers = [node.tier for node in result.nodes]
    assert 1 <= tiers.count("central") <= 3
    assert "secondary" in tiers
    assert all(hasattr(edge, "subject") and hasattr(edge, "object") for edge in result.edges)
    node_tier = {node.name: node.tier for node in result.nodes}

    first_stage_children = [
        edge
        for edge in result.edges
        if node_tier.get(edge.subject) == "central" and node_tier.get(edge.object) == "secondary"
    ]
    per_central = {}
    for edge in first_stage_children:
        per_central[edge.subject] = per_central.get(edge.subject, 0) + 1
    assert all(3 <= count <= 6 for count in per_central.values())

    second_stage_parents = {
        edge.subject
        for edge in result.edges
        if node_tier.get(edge.subject) == "secondary" and node_tier.get(edge.object) == "secondary"
    }
    assert len(second_stage_parents) <= 5
    final_stage_parents = {
        edge.subject
        for edge in result.edges
        if node_tier.get(edge.subject) == "secondary" and node_tier.get(edge.object) == "periphery"
    }
    assert len(final_stage_parents) <= 4
    central_links = [
        edge
        for edge in result.edges
        if node_tier.get(edge.subject) == "central"
        and node_tier.get(edge.object) == "central"
        and {edge.subject, edge.object} == {"c1", "c2"}
    ]
    assert len(central_links) == 1


def test_build_subgraph_v2_stage1_minimum_with_sparse_direct_neighbours(monkeypatch):
    graph = nx.DiGraph()
    graph.add_node("c1", description="central", weight=4.0)
    graph.add_node("near1", description="near 1", weight=2.0)
    graph.add_node("via", description="bridge", weight=2.0)
    graph.add_node("deep1", description="deep node 1", weight=1.5)
    graph.add_node("deep2", description="deep node 2", weight=1.4)

    graph.add_edge("c1", "near1", predicate="related", description="c1-near1", weight=2.0)
    graph.add_edge("c1", "via", predicate="related", description="c1-via", weight=2.0)
    graph.add_edge("via", "deep1", predicate="branch", description="via-deep1", weight=1.8)
    graph.add_edge("via", "deep2", predicate="branch", description="via-deep2", weight=1.7)

    async def fake_find_central_nodes(client, graph_arg, query):
        return ["c1"]

    monkeypatch.setattr(graph_v2, "_find_central_nodes", fake_find_central_nodes)
    config = BranchingConfig(
        first_secondary_min=3,
        first_secondary_max=3,
        second_parents_max=0,
        final_parents_max=0,
    )
    result = asyncio.run(
        graph_v2.build_subgraph_v2(
            client=object(),
            graph=graph,
            query="test query",
            config=config,
        )
    )

    node_tier = {node.name: node.tier for node in result.nodes}
    stage1_edges = [
        edge
        for edge in result.edges
        if edge.subject == "c1"
        and node_tier.get(edge.subject) == "central"
        and node_tier.get(edge.object) == "secondary"
    ]
    assert len(stage1_edges) >= 3


def test_build_subgraph_v2_colour_inheritance(monkeypatch):
    graph = _make_synthetic_graph()

    async def fake_find_central_nodes(client, graph_arg, query):
        return ["c1", "c2"]

    monkeypatch.setattr(graph_v2, "_find_central_nodes", fake_find_central_nodes)
    config = BranchingConfig(
        first_secondary_min=3,
        first_secondary_max=3,
        second_parents_max=2,
        second_secondary_min=2,
        second_secondary_max=2,
        final_parents_max=1,
        final_periphery_min=2,
        final_periphery_max=2,
    )
    result = asyncio.run(
        graph_v2.build_subgraph_v2(
            client=object(),
            graph=graph,
            query="energy access",
            config=config,
        )
    )

    node_by_name = {node.name: node for node in result.nodes}
    for edge in result.edges:
        subject = node_by_name[edge.subject]
        object_node = node_by_name[edge.object]
        if object_node.tier in {"secondary", "periphery"} and subject.tier in {"secondary", "periphery"}:
            assert object_node.colour == subject.colour


def test_find_central_nodes_keeps_single_when_other_hits_are_weak():
    graph = nx.DiGraph()
    graph.add_node("A")
    graph.add_node("B")
    graph.add_node("C")
    rows = [
        {"name": "A", "_distance": 0.08},
        {"name": "B", "_distance": 0.44},
        {"name": "C", "_distance": 0.61},
    ]
    client = _FakeClient(rows)

    selected = asyncio.run(
        graph_v2._find_central_nodes(
            client,
            graph,
            "long query with many terms about energy policy transition and planning",
        )
    )

    assert selected == ["A"]


def test_find_central_nodes_allows_multiple_when_top_hits_are_close():
    graph = nx.DiGraph()
    graph.add_node("A")
    graph.add_node("B")
    graph.add_node("C")
    rows = [
        {"name": "A", "_distance": 0.08},
        {"name": "B", "_distance": 0.10},
        {"name": "C", "_distance": 0.12},
    ]
    client = _FakeClient(rows)

    selected = asyncio.run(
        graph_v2._find_central_nodes(
            client,
            graph,
            "long query with many terms about energy policy transition and planning",
        )
    )

    assert selected == ["A", "B", "C"]


def test_pick_diverse_blocks_near_synonyms_already_selected_globally():
    candidates = [
        graph_v2.Candidate(
            node_name="ADAPTATION",
            parent="CLIMATE FINANCE",
            predicate="relates_to",
            description=None,
            weight=1.0,
            score=0.9,
            domain="finance",
        ),
        graph_v2.Candidate(
            node_name="CLIMATE RESILIENCE",
            parent="CLIMATE FINANCE",
            predicate="relates_to",
            description=None,
            weight=1.0,
            score=0.8,
            domain="risk",
        ),
    ]
    blocked = {"CLIMATE CHANGE ADAPTATION"}
    selected = graph_v2._pick_diverse(
        candidates=candidates,
        target=2,
        blocked=blocked,
        parent_name="CLIMATE FINANCE",
    )

    names = {item.node_name for item in selected}
    assert "ADAPTATION" not in names
    assert "CLIMATE RESILIENCE" in names
