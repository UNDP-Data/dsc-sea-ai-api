"""
Utilities for processing and transforming data, but it also includes colour definitions.
"""

import networkx as nx

__all__ = ["get_node_metadata"]

GRAYS = ["#232E3D", "#55606E", "#A9B1B7", "#D4D6D8", "#EDEFF0"]
GREENS = ["#006A51", "#2C8956", "#56A75A", "#81C55F", "#C1DD85"]
REDS = ["#aa1d09", "#d9382d", "#e47559", "#eea57d", "#f6d39e"]
YELLOWS = ["#B59005", "#FBC412", "#FFEB00", "#FFF27A", "#FFFAAA"]
BLUES = ["#095aab", "#347cbc", "#5f9dcc", "#89bfdc", "#bae0e1"]
PALLETES = [GRAYS, GREENS, REDS, YELLOWS, BLUES]


def get_node_metadata(
    nodes: list[str],
    edges: list[tuple[str, str]],
    source: str,
) -> dict[str, dict]:
    """
    Get node metadata such as neighbourhood position and colours from a graph.

    Parameters
    ----------
    nodes : list[str]
        List of node names.
    edges : list[tuple[str, str]]
        List of edges in the form (source, target).
    source : str
        Name of the source node to find paths from, i.e., the central node.

    Returns
    -------
    dict[str, dict]
        Mapping from node names to metadata dictionary.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    graph.nodes[source]["neighbourhood"] = 0
    graph.nodes[source]["colour"] = "#9F7DC5"
    # there may be more than one shortest path, but any one works
    for _, (path, *_) in nx.single_source_all_shortest_paths(graph, source=source):
        # skip the source node as it is already processed
        for hop, node_name in enumerate(path[1:], start=1):
            graph.nodes[node_name]["neighbourhood"] = hop
            # colours for nodes in a path are determined by 1-hop neighbours
            if hop == 1:
                # deterministically choose a palette based on a node's name
                index = sum(map(ord, node_name)) % len(PALLETES)
            # reuse the same palette for all nodes along the path but use a different shade
            graph.nodes[node_name]["colour"] = PALLETES[index][hop - 1]
    return dict(graph.nodes(data=True))
