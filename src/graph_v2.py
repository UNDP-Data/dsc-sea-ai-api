"""
Backward-compatible adapter for the V2 staged graph builder.

This module keeps legacy import paths stable while delegating implementation
to `src.kg.v2`.
"""

from __future__ import annotations

from .kg import v2 as _impl

# Public classes/types preserved for compatibility.
BranchingConfig = _impl.BranchingConfig
Candidate = _impl.Candidate

# Private helpers used by tests; monkeypatching these names should still work.
_find_central_nodes = _impl._find_central_nodes
_pick_diverse = _impl._pick_diverse


def _sync_test_overrides() -> None:
    """
    Propagate patched helper refs from this compatibility module into impl.
    """
    _impl._find_central_nodes = _find_central_nodes
    _impl._pick_diverse = _pick_diverse


async def build_subgraph_v2(*args, **kwargs):
    """
    Delegate to the real V2 builder, while honoring test monkeypatches.
    """
    _sync_test_overrides()
    return await _impl.build_subgraph_v2(*args, **kwargs)


__all__ = [
    "BranchingConfig",
    "Candidate",
    "_find_central_nodes",
    "_pick_diverse",
    "build_subgraph_v2",
]
