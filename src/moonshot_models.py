"""Request and response models for the Moonshot assistant API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MoonshotHealthResponse(BaseModel):
    ok: bool = True
    configured: bool
    provider: str | None = None
    parseModel: str | None = None
    synopsisModel: str | None = None


class ParseQueryRequest(BaseModel):
    query: str
    locale: str = "en"
    filterCatalog: dict[str, Any] = Field(default_factory=dict)


class ParseQueryResponse(BaseModel):
    filters: dict[str, str] = Field(default_factory=dict)
    unresolvedTerms: list[str] = Field(default_factory=list)


class ProjectSynopsisRequest(BaseModel):
    query: str
    locale: str = "en"
    filters: dict[str, str] = Field(default_factory=dict)
    summaryMetrics: dict[str, Any] = Field(default_factory=dict)
    projectContext: dict[str, Any] = Field(default_factory=dict)


class ProjectSynopsisResponse(BaseModel):
    synopsis: str
