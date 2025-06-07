"""
Utilities for processing and transforming data, but it also includes colour definitions.
"""

from typing import Iterable

import networkx as nx

__all__ = ["extract_node_positions", "get_node_colours"]

GRAYS = ["#232E3D", "#55606E", "#A9B1B7", "#D4D6D8", "#EDEFF0"]
GREENS = ["#006A51", "#2C8956", "#56A75A", "#81C55F", "#C1DD85"]
REDS = ["#aa1d09", "#d9382d", "#e47559", "#eea57d", "#f6d39e"]
YELLOWS = ["#B59005", "#FBC412", "#FFEB00", "#FFF27A", "#FFFAAA"]
BLUES = ["#095aab", "#347cbc", "#5f9dcc", "#89bfdc", "#bae0e1"]
PALLETES = [GRAYS, GREENS, REDS, YELLOWS, BLUES]


def extract_node_positions(neighbourhoods: dict[int, Iterable[str]]) -> dict[str, int]:
    """
    Extract node position in the neighbourhood.

    Since there may be more than one path from a central node A to
    some other node B – e.g., if the edges are A-B, A-C, C-B – the
    closest position to the central node is kept.

    Parameters
    ----------
    neighbourhoods : dict[int, Iterable[str]]
        Mapping from neighbourhood indices to an iterable of nodes.

    Returns
    -------
    dict[str, int]
        Mapping from node names to their corresponding neighbourhood indices.
    """
    positions = {}
    # traverse from the closest neighbours
    for hop, node_names in sorted(neighbourhoods.items(), key=lambda x: x[0]):
        for node_name in node_names:
            # keep only the position closest to the central node
            if node_name not in positions:
                positions[node_name] = hop
    return positions


def get_node_colours(graph: nx.Graph, source: str) -> dict[str, str]:
    """
    Deterministically get colours for nodes from a graph.

    Parameters
    ----------
    G : nx.Graph
        A graph to traverse.
    source : str
        Name of the source node to find paths from, i.e., the central node.

    Returns
    -------
    dict[str, str]
        Mapping from node names to hex colour values.
    """
    graph.nodes[source]["colour"] = "#9F7DC5"
    # there may be more than one shortest path, but any one works
    for _, (path, *_) in nx.single_source_all_shortest_paths(graph, source=source):
        # skip the source node as it is already processed
        for hop, node_name in enumerate(path[1:], start=1):
            # colours for nodes in a path are determined by 1-hop neighbours
            if hop == 1:
                # deterministically choose a palette based on a node's name
                index = sum(map(ord, node_name)) % len(PALLETES)
            # reuse the same palette for all nodes along the path but use a different shade
            graph.nodes[node_name]["colour"] = PALLETES[index][hop - 1]
    return {name: data.get("colour") for name, data in graph.nodes(data=True)}
