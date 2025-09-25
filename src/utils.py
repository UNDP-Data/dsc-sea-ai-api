"""
Utilities for processing and transforming data, but it also includes colour definitions.
"""

import networkx as nx

__all__ = ["get_neighbourhood_nodes", "get_closest_nodes", "prune_edges"]

GRAYS = ["#232E3D", "#55606E", "#A9B1B7", "#D4D6D8", "#EDEFF0"]
GREENS = ["#006A51", "#2C8956", "#56A75A", "#81C55F", "#C1DD85"]
REDS = ["#aa1d09", "#d9382d", "#e47559", "#eea57d", "#f6d39e"]
YELLOWS = ["#B59005", "#FBC412", "#FFEB00", "#FFF27A", "#FFFAAA"]
BLUES = ["#095aab", "#347cbc", "#5f9dcc", "#89bfdc", "#bae0e1"]
PALLETES = [GRAYS, GREENS, REDS, YELLOWS, BLUES]


def get_neighbourhood_nodes(
    graph: nx.Graph, sources: list[str], hops: int = 3
) -> list[str]:
    """
    Find nodes within a k-hop neighbourhood from source nodes in the graph.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph.
    sources : list[str]
        List of node names to find shortest paths from.
    hops : int, default=3
        Keep nodes that are within hops radius.

    Returns
    -------
    list[str]
        Names of the nodes that are within hops neighbourhood.
    """
    nodes = set()
    for source in sources:
        paths = nx.single_source_shortest_path_length(graph, source=source, cutoff=hops)
        for node, _ in paths.items():
            nodes.add(node)
    return nodes


def get_closest_nodes(graph: nx.Graph, sources: list[str], n: int = 30) -> list[str]:
    """
    Find nodes top `n` closest nodes from source nodes in the graph.

    This function assumes that edge weight denotes its importance and this uses
    its reciprocal for computing distances on shortest paths.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph.
    sources : list[str]
        List of node names to find shortest paths from.
    n : int, default=30
        Maximum number of closest nodes to return.

    Returns
    -------
    list[str]
        Names of the nodes closest to the sources.
    """
    distances, _ = nx.multi_source_dijkstra(
        graph,
        sources=sources,
        # use reciprocal of weight for computing distance
        weight=lambda u, v, d: 1 / d["weight"],
    )
    nodes = [node for node, _ in sorted(distances.items(), key=lambda x: (x[1], x[0]))]
    return nodes[:n]


def prune_edges(graph: nx.Graph, n: int = 100) -> list[str]:
    """
    Prune edges in the graph to keep `n` edges with the highest weight.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph.
    n : int, default=100
        Maximum number of edges to keep.

    Returns
    -------
    nx.Graph
        The input graph with pruned nodes.
    """
    edges = sorted(
        graph.edges(data=True), key=lambda edge: edge[2]["weight"], reverse=True
    )
    graph.clear_edges()
    graph.add_edges_from(edges[:n])
    return graph
