"""
Entities (models) and related routines to define the data layer.
"""

from typing import Literal

from lancedb.pydantic import LanceModel
from pydantic import BaseModel, Field

__all__ = ["Node", "Edge", "Graph", "Document", "Message", "AssistantResponse"]


class Node(BaseModel):
    """
    A node in the knowledge graph representing a single concept.
    """

    name: str = Field(
        description="Name of the node",
        examples=["climate change mitigation", "energy policy"],
    )
    description: str | None = Field(
        description="Description of the node",
        examples=[
            "Actions to limit global warming by reducing greenhouse gas emissions.",
            "Rules managing energy use and resources sustainably.",
        ],
    )
    weight: int | None = Field(
        default=1,
        description="Numeric value indicating the node's importance on a 5-point scale",
        examples=[3, 4],
        ge=1,
        le=5,
    )
    metadata: dict = Field(description="Arbitrary metadata about the node")

    def __hash__(self) -> int:
        return self.name.__hash__()


class Edge(BaseModel):
    """
    An edge in the knowledge graph representing a relation between nodes.

    Note that a node for `subject` property is guaranteed to exists, but the `object`
    value might not have a corresponding node entity in the graph.
    """

    subject: str = Field(
        description="Name of the node the edge goes from",
        examples=["climate change mitigation"],
    )
    predicate: str = Field(
        description="Name of the edge relation",
        examples=["addresses"],
    )
    object: str = Field(
        description="Name of the node the edge goes to",
        examples=["energy policy"],
    )
    description: str = Field(
        description="Description of the edge",
        examples=[
            "Climate adaptation strategies address the impacts of extreme weather events on communities and infrastructure.",
            "Climate adaptation strategies support community engagement to ensure inclusive decision-making and implementation.",
        ],
    )
    weight: int = Field(
        default=1,
        description="Numeric value indicating the edge's importance on a 5-point scale",
        examples=[3, 4],
        ge=1,
        le=5,
    )
    level: int = Field(
        description="Numeric values indication the edge's level on a 3-point scale",
        examples=[1, 2],
        ge=1,
        le=5,
    )

    def __hash__(self) -> int:
        return (self.subject + self.predicate + self.object).__hash__()


class Graph(BaseModel, frozen=True):
    """
    Knowledge graph centred around a single concept.
    """

    nodes: list[Node] = Field(description="A list of nodes in the graph.")
    edges: list[Edge] = Field(description="A list of edges in the graph.")

    def __add__(self, other: "Graph") -> "Graph":
        if not isinstance(other, Graph):
            raise NotImplementedError
        nodes = set(self.nodes) | set(other.nodes)
        edges = set(self.edges) | set(other.edges)
        return Graph(nodes=nodes, edges=edges)


class Document(LanceModel):
    """
    Publication document.
    """

    title: str | None = Field(description="Document title if available")
    year: int | None = Field(description="Publication year if available")
    language: str = Field(description="Document language")
    url: str = Field(description="URL to the source document")
    summary: str | None = Field(description="Brief document summary if available")


class Message(BaseModel):
    role: Literal["assistant", "human"] = Field(
        description="The actor the message belongs to",
        examples=["human"],
    )
    content: str = Field(
        description="Text content of the message",
        min_length=8,
        max_length=16_384,
        examples=[
            "How does climate change adaptation differ from climate change mitigation?"
        ],
    )


class AssistantResponse(Message):
    """
    Assistant message from the LLM, optionally with relevant knowledge subgraphs.
    """

    ideas: list[str] | None = Field(
        description="A list of relevant query ideas based on the user message"
    )
    documents: list[Document] = Field(
        description="One or more documents relevant to the user message"
    )
    graph: Graph | None = Field(default=None)
