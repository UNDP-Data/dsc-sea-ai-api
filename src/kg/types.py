"""
Knowledge graph response models (versioned APIs).
"""

from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "GraphV2Parameters",
    "NodeV2",
    "EdgeV2",
    "GraphV2",
]


class GraphV2Parameters(BaseModel):
    """
    Query parameters for `/graph/v2`.
    """

    query: str = Field(
        description="A query to retrieve a staged knowledge graph for",
        min_length=2,
        json_schema_extra={"example": "climate change mitigation"},
    )


class NodeV2(BaseModel):
    """
    V2 node model with explicit tier.
    """

    name: str = Field(description="Name of the node")
    description: str | None = Field(default=None, description="Node description")
    tier: Literal["central", "secondary", "periphery"] = Field(
        description="Tier in staged graph expansion"
    )
    weight: float = Field(default=0.0, ge=0.0, description="Node relevance weight")
    colour: str = Field(
        default="#6B7280",
        description="Node colour (dark rainbow branch colour for non-central nodes)",
    )


class EdgeV2(BaseModel):
    """
    V2 edge model.
    """

    subject: str = Field(description="Name of the source node")
    predicate: str = Field(default="related_to", description="Relation predicate")
    object: str = Field(description="Name of the target node")
    description: str | None = Field(default=None, description="Relation description")
    weight: float = Field(default=1.0, ge=0.0, description="Edge relevance weight")


class GraphV2(BaseModel):
    """
    V2 staged graph response.
    """

    nodes: list[NodeV2] = Field(description="A list of graph nodes.")
    edges: list[EdgeV2] = Field(description="A list of graph edges.")
