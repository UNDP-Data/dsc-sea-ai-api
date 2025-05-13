"""
Entities (models) and related routines to define the data layer.
"""

from pydantic import BaseModel, Field

__all__ = [
    "BaseConcept",
    "Concept",
    "Relation",
    "Subgraph",
    "KnowledgeGraph",
    "HumanMessage",
    "AssistantMessage",
]


class BaseConcept(BaseModel):
    """
    Base class for concept nodes in the knowledge graph. Also used for subconcepts.
    """

    name: str = Field(
        alias="Entity",
        description="Name of the concept",
        examples=["climate change mitigation", "energy policy"],
    )
    description: str | None = Field(
        alias="Description",
        description="Description of the concept",
        examples=[
            "Actions to limit global warming by reducing greenhouse gas emissions.",
            "Rules managing energy use and resources sustainably.",
        ],
    )
    importance: int | None = Field(
        default=1,
        alias="Importance",
        description="Numeric value indicating the concept's importance on a 5-point scale",
        examples=[3, 4],
        ge=1,
        le=5,
    )

    @classmethod
    def from_subelement(cls, data: dict) -> "BaseConcept":
        """
        Class method to create a BaseConcept from Sub-element relation dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing sub-element data.

        Returns
        -------
        BaseConcept
            Base concept object populated with the data.
        """
        data = data.copy()
        return cls(Entity=data.pop("Sub-element"), **data)


class Concept(BaseConcept):
    """
    A basic concept node in the knowledge graph.
    """

    category: str | None = Field(
        alias="Category",
        description="One of the predefined categories the concept belongs to",
        examples=["Crisis", "Policy"],
    )
    tags: list[str] | None = Field(
        alias="Tags",
        description="One or more relevant tags from an open vocabulary",
        examples=[["Adaptation", "Africa"], ["Biomass"]],
    )
    dimension: str | None = Field(
        alias="Dimension",
        description="One of the predefined dimensions the concept is relevant to",
        examples=["Technology", "Sustainable Development"],
    )
    acronym: str | None = Field(
        default=None,
        description="Acronym used to refer to the concept if applicable",
        examples=["NDCs", "SIDS"],
    )
    synonyms: list[str] | None = Field(
        default=None,
        alias="Synonyms",
        description="One or more relevant synonyms related to the concept if applicable",
        examples=[["Emissions reduction"], ["Energy governance", "Energy regulation"]],
    )
    subconcepts: list[BaseConcept] | None = Field(
        default=None,
        description="One or more relevant subconcepts if applicable",
    )


class Relation(BaseModel):
    """
    A basic relation edge in the knowledge graph.
    """

    name: str = Field(
        alias="Relation",
        description="Name of the relation",
        examples=["addresses", "supports"],
    )
    description: str = Field(
        alias="Description",
        description="Description of the relation",
        examples=[
            "Climate adaptation strategies address the impacts of extreme weather events on communities and infrastructure.",
            "Climate adaptation strategies support community engagement to ensure inclusive decision-making and implementation.",
        ],
    )
    importance: int = Field(
        default=1,
        alias="Importance",
        description="Numeric value indicating the relation's importance on a 5-point scale",
        examples=[3, 4],
        ge=1,
        le=5,
    )
    level: int = Field(
        description="Numeric values indication the relation's level on a 3-point scale",
        examples=[1, 2],
        ge=1,
        le=3,
    )
    to_concept: str = Field(
        alias="Object",
        description="Name of the concept the relation applies to",
        examples=["extreme weather events", "community engagement"],
    )


class Subgraph(BaseModel):
    """
    Knowledge graph for a single concept.
    """

    concept: Concept = Field(description="The main concept of the subgraph")
    relations: list[Relation] = Field(description="Relations with other concepts")

    @classmethod
    def from_kg(cls, data: dict) -> "Subgraph":
        """
        Class method to create a knowledge grapg for a single concept from KG data.

        Parameters
        ----------
        data : dict
            Dictionary containing knowledge graph data for a single concept.

        Returns
        -------
        Subgraph
            Knowledge graph for a single concept.
        """
        kg = data["knowledge graph"]
        subconcepts = map(BaseConcept.from_subelement, kg["subelement_relations"])
        concept = Concept(**data["metadata"], subconcepts=subconcepts)
        relations = []
        for level, edges in kg["relations"].items():
            relations.extend([Relation(**edge, level=int(level[-1])) for edge in edges])
        return cls(concept=concept, relations=relations)


class KnowledgeGraph(BaseModel):
    subgraphs: list[Subgraph]


class HumanMessage(BaseModel):
    """
    User message for the LLM.
    """

    content: str = Field(
        description="Text content of the message",
        min_length=8,
        max_length=1024,
        examples=[
            "How does climate change adaptation differ from climate change mitigation?"
        ],
    )


class AssistantMessage(BaseModel):
    """
    Assistant message from the LLM, optionally with relevant knowledge subgraphs.
    """

    content: str = Field(description="Text content of the message")
    entities: list[str] | None = Field(
        description="A list of relevant entities extracted from the user message"
    )
    ideas: list[str] | None = Field(
        description="A list of relevant query ideas based on the user message"
    )
    excerpts: dict = Field(
        description="A dictionary of text excerpts relevant to the user message"
    )
    subgraphs: list[Subgraph] | None = Field(default=None)
