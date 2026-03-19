"""
V1 graph service wrapper.
"""

from __future__ import annotations

import networkx as nx

from ..database import Client
from ..entities import Graph


async def build_subgraph_v1(
    client: Client,
    graph: nx.Graph,
    query: str | list[str],
    hops: int = 2,
) -> Graph:
    """
    Build a V1 subgraph using the legacy client path.
    """
    return await client.find_subgraph(graph=graph, query=query, hops=hops)
