"""
Knowledge graph versioned services.
"""

from __future__ import annotations

__all__ = ["build_subgraph_v1", "build_subgraph_v2", "BranchingConfig"]


def __getattr__(name: str):
    """
    Lazy-load versioned KG exports to avoid import cycles at startup.
    """
    if name == "build_subgraph_v1":
        from .v1 import build_subgraph_v1

        return build_subgraph_v1
    if name == "build_subgraph_v2":
        from .v2 import build_subgraph_v2

        return build_subgraph_v2
    if name == "BranchingConfig":
        from .v2 import BranchingConfig

        return BranchingConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
