"""
Utilities for processing and transforming data.
"""

from typing import Iterable

__all__ = ["extract_node_positions"]


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
