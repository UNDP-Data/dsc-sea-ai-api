"""
Entities (models) and related routines to define the data layer.
"""

from enum import Enum, auto
from typing import Literal

from lancedb.pydantic import LanceModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

__all__ = [
    "SearchMethod",
    "GraphParameters",
    "Node",
    "Edge",
    "Graph",
    "Document",
    "Chunk",
    "Message",
    "AssistantResponse",
]


class SearchMethod(Enum):
    """
    Search method for graph retrieval.
    """

    EXACT = auto()
    VECTOR = auto()


class SharedParameters(BaseModel):
    """
    Shared query parameters for the underlying knowledge graph.
    """

    hops: int = Field(
        default=2,
        ge=0,
        description="Number of hops to extract nodes from."
        "For example, 1 means extract the central node and the immediate neighbours"
        " (1-hop neighbourhood).",
    )


class GraphParameters(SharedParameters):
    """
    Query parameters for `/graph` endpoint.
    """

    query: str = Field(
        description="A query to retrieve a knowledge graph for",
        min_length=2,
        json_schema_extra={"example": "climate change mitigation"},
    )


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
    neighbourhood: int = Field(
        default=0,
        description="""Numeric value for a k-hop neighbourhood, with designations as follows:
        0 for the central node(s), 1 for secondary nodes and so on""",
        examples=[0, 2],
        ge=0,
    )
    weight: float = Field(
        default=0.0,
        description="Numeric value indicating the node's relevance",
        examples=[2.71828, 3.14159],
        ge=0.0,
    )
    colour: str = Field(
        default="#A9B1B7",
        description="Hex colour value for the node",
        examples=["#3288CE", "#55606E"],
    )
    metadata: dict = Field(description="Arbitrary metadata about the node")

    def __hash__(self) -> int:
        return self.name.__hash__()

    def __lt__(self, other):
        if isinstance(other, Node):
            return self.name < other.name
        return NotImplemented


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
            "Climate adaptation strategies address the impacts of extreme weather"
            " events on communities and infrastructure.",
            "Climate adaptation strategies support community engagement to ensure"
            " inclusive decision-making and implementation.",
        ],
    )
    weight: float = Field(
        default=1.0,
        description="Numeric value indicating the edge's importance",
        examples=[2.71828, 3.14159],
        ge=0.0,
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

    title: str = Field(description="Document title if available")
    year: int = Field(description="Publication year if available")
    language: str = Field(description="Document language")
    url: str = Field(description="URL to the source document")
    summary: str | None = Field(description="Brief document summary if available")

    def __hash__(self) -> int:
        """
        Allows Document instances to be hashable based on the title.
        """
        return hash(self.title)

    def __eq__(self, other: "Document") -> bool:
        """
        Compares two Document instances for equality based only on the title.
        """
        if not isinstance(other, Document):
            return NotImplemented
        return self.title == other.title

    def __lt__(self, other: "Document") -> bool:
        """
        Less-than comparison for sorting by year first, then title.
        """
        if self.year != other.year:
            # descending order by year
            return self.year > other.year
        # alphabetically by title (if years are the same)
        return self.title < other.title


class Chunk(Document):
    """
    Chunk of a publication document.
    """

    content: str = Field(description="Text content of a chunk")

    def to_context(self) -> dict:
        """
        Convert a chunk to a simple context to be fed to a GenAI model.
        """
        return {
            "content": self.content,
            "source": f"[{self.title} ({self.year})]({self.url})",
        }


class Message(BaseModel):
    """
    Simple message for handling conversations with a chatbot.
    """

    role: Literal["assistant", "human"] = Field(
        description="The actor the message belongs to",
        examples=["human"],
    )
    content: str = Field(
        description="Text content of the message",
        min_length=0,
        max_length=16_384,
        examples=[
            "How does climate change adaptation differ from climate change mitigation?"
        ],
    )

    def to_langchain(self) -> AIMessage | HumanMessage:
        """
        Convert the message to a langchain-compatible class.

        Returns
        -------
        AIMessage | HumanMessage
            Langchain message class for AI or human.
        """
        return (
            AIMessage(self.content)
            if self.role == "assistant"
            else HumanMessage(self.content)
        )


class AssistantResponse(Message):
    """
    Assistant message from the LLM, optionally with relevant knowledge subgraphs.
    """

    ideas: list[str] | None = Field(
        default=None,
        description="A list of relevant query ideas based on the user message",
    )
    documents: list[Document] | None = Field(
        default=None, description="One or more documents relevant to the user message"
    )
    graph: Graph | None = Field(default=None)
