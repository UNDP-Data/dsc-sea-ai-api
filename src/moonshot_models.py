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
    query: str = Field(min_length=1, max_length=500)
    locale: str = Field(default="en", max_length=32)
    filterCatalog: dict[str, Any] = Field(default_factory=dict)


class ParseQueryResponse(BaseModel):
    filters: dict[str, str] = Field(default_factory=dict)
    unresolvedTerms: list[str] = Field(default_factory=list)


class ProjectSynopsisRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    locale: str = Field(default="en", max_length=32)
    filters: dict[str, str] = Field(default_factory=dict)
    summaryMetrics: dict[str, Any] = Field(default_factory=dict)
    projectContext: dict[str, Any] = Field(default_factory=dict)


class ProjectSynopsisResponse(BaseModel):
    synopsis: str
