"""
Routines for database operations for RAG.
"""

import asyncio
import contextlib
import json
import logging
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from inspect import isawaitable
from time import monotonic
from urllib.parse import urlparse

import lancedb
import networkx as nx
import pandas as pd
from langchain_core.tools import tool

from . import corpus, genai, utils
from .entities import Chunk, Document, DocumentRecord, Graph, Node, SearchMethod, SourceRecord

__all__ = ["get_storage_options", "get_connection", "Client", "retrieve_chunks"]
logger = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[a-z0-9]+")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
DEFAULT_DOCUMENT_CACHE_TTL_SECONDS = 300.0
DOCUMENT_INDEX_FIELDS = [
    "document_id",
    "source_id",
    "canonical_title",
    "subtitle",
    "url",
    "language",
    "document_type",
    "publication_date",
    "year",
    "summary",
    "status",
    "publisher",
    "series_name",
    "topic_tags",
    "topic_tags_text",
    "country_codes",
    "region_codes",
    "geography_tags_text",
    "audience_tags",
    "audience_tags_text",
    "source_priority",
    "quality_score",
    "authority_tier",
    "is_flagship",
    "is_data_report",
]
_DOCUMENT_INDEX_CACHE: dict[str, object] = {
    "rows": None,
    "loaded_at": 0.0,
    "connection_id": None,
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "more",
    "of",
    "on",
    "or",
    "please",
    "provide",
    "tell",
    "that",
    "the",
    "this",
    "to",
    "what",
    "with",
}
EXPLANATORY_PREFIXES = (
    "tell me more about",
    "tell me about",
    "what is",
    "what are",
    "explain",
    "provide more context",
    "give more context",
    "please provide more context",
)
RELATIONSHIP_MARKERS = {
    "connection",
    "connections",
    "relation",
    "relations",
    "relationship",
    "relationships",
    "link",
    "links",
    "between",
}
DATA_MARKERS = (
    "latest",
    "current",
    "recent",
    "most recent",
    "as of",
    "tracking",
    "indicator",
    "indicators",
    "data",
    "trend",
    "trends",
    "progress",
    "status",
    "rate",
    "rates",
    "share",
    "shares",
    "statistics",
    "statistical",
    "population",
    "how many",
    "how much",
    "percent",
    "percentage",
)
POLICY_MARKERS = (
    "policy",
    "policies",
    "tariff",
    "tariffs",
    "fit",
    "feed in tariff",
    "feed-in tariff",
    "subsidy",
    "subsidies",
    "regulation",
    "regulations",
    "regulatory",
    "governance",
    "incentive",
    "incentives",
)
IMPLEMENTATION_MARKERS = (
    "implementation",
    "deploy",
    "deployment",
    "program",
    "programme",
    "project",
    "case study",
    "example",
    "examples",
    "infrastructure",
    "mini-grid",
    "mini grids",
    "off-grid",
)
DEFINITION_MARKERS = (
    "what is ",
    "what are ",
    "define ",
    "definition",
    "meaning of",
)
GENERIC_REPORT_PATTERNS = (
    "tracking sdg7 report",
    "tracking sdg 7 report",
    "report",
)
FLAGSHIP_REPORT_PATTERNS = (
    "tracking sdg7",
    "tracking sdg 7",
    "progress toward sustainable energy",
    "global tracking framework",
)
GLOSSARY_PATTERNS = (
    "dictionary",
    "glossary",
    "lexicon",
    "terminology",
)
POLICY_DOCUMENT_PATTERNS = (
    "policy brief",
    "action brief",
    "toolkit",
    "guidance",
    "guidance note",
    "framework",
    "roadmap",
    "governance",
)
CASE_STUDY_PATTERNS = (
    "case study",
    "assessment",
    "programme",
    "program",
    "initiative",
    "nama",
)
QUERY_REGION_ALIASES: dict[str, tuple[str, ...]] = {
    "africa": ("africa", "sub saharan africa", "sub-saharan africa"),
    "asia": ("asia", "south asia", "southeast asia", "central asia"),
    "latin america": (
        "latin america",
        "latin america and the caribbean",
        "latin america & the caribbean",
        "caribbean",
        "lac",
    ),
    "europe": ("europe", "european union"),
    "middle east": ("middle east", "mena", "arab states"),
    "small island developing states": ("sids", "small island developing states"),
    "global": ("global", "worldwide", "international"),
}
QUERY_COUNTRY_ALIASES: dict[str, tuple[str, ...]] = {
    "NGA": ("nigeria",),
    "COD": ("democratic republic of congo", "dr congo", "drc"),
    "ETH": ("ethiopia",),
    "KEN": ("kenya",),
    "IND": ("india",),
    "IDN": ("indonesia",),
    "BRA": ("brazil",),
}
COUNTRY_TO_REGION: dict[str, str] = {
    "NGA": "africa",
    "COD": "africa",
    "ETH": "africa",
    "KEN": "africa",
    "IND": "asia",
    "IDN": "asia",
    "BRA": "latin america",
}


def _env_timeout_seconds(
    name: str,
    default: float,
    *,
    min_value: float = 0.1,
) -> float:
    """
    Parse timeout configuration from environment with safe fallback.

    Invalid, empty, or too-small values fall back to the provided default.
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        logger.warning(
            "Invalid %s value %r; using default=%s",
            name,
            raw,
            default,
        )
        return default
    if parsed < min_value:
        logger.warning(
            "Ignoring %s value %s below minimum %s; using default=%s",
            name,
            parsed,
            min_value,
            default,
        )
        return default
    return parsed


@dataclass(frozen=True)
class RetrievalProfile:
    query: str
    normalized_query: str
    query_tokens: set[str]
    query_phrases: tuple[str, ...]
    explicit_years: list[int]
    prefer_recent: bool
    explanatory: bool
    intent: str
    country_scopes: frozenset[str]
    region_scopes: frozenset[str]
    fallback_region_scopes: frozenset[str]
    has_geographic_scope: bool


@dataclass(frozen=True)
class DocumentCandidate:
    key: str
    series_id: str
    source_domain: str
    source_org: str
    document_type: str
    score: float
    topicality: float
    rows: tuple[tuple[dict, float], ...]


@dataclass(frozen=True)
class DocumentHit:
    row: dict
    score: float


def _extract_focus_phrases(query: str | None) -> tuple[str, ...]:
    trimmed = " ".join((_trim_query_for_retrieval(query or "")).lower().split())
    if not trimmed:
        return ()
    phrases: list[str] = []
    if " and " in trimmed:
        for part in trimmed.split(" and "):
            part = " ".join(
                token
                for token in TOKEN_RE.findall(part.lower())
                if token not in STOPWORDS and token not in RELATIONSHIP_MARKERS
            )
            if len(part.split()) >= 2 and part not in phrases:
                phrases.append(part)
    if not phrases:
        normalized_trimmed = _normalize_text(trimmed)
        if len(normalized_trimmed.split()) >= 2:
            phrases.append(normalized_trimmed)
    return tuple(phrases[:3])


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(TOKEN_RE.findall(text.lower()))


def _tokenize_text(text: str | None) -> set[str]:
    return {
        token
        for token in TOKEN_RE.findall((text or "").lower())
        if token not in STOPWORDS and len(token) > 1
    }


def _ordered_query_tokens(text: str | None) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall((text or "").lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def _ordered_raw_tokens(text: str | None) -> list[str]:
    return [token for token in TOKEN_RE.findall((text or "").lower()) if len(token) > 1]


def _extract_query_phrases(text: str | None, max_words: int = 4) -> tuple[str, ...]:
    tokens = _ordered_query_tokens(text)
    phrases: list[str] = []
    for size in range(min(max_words, len(tokens)), 1, -1):
        for start in range(0, len(tokens) - size + 1):
            phrase = " ".join(tokens[start : start + size])
            if phrase not in phrases:
                phrases.append(phrase)
    return tuple(phrases)


def _extract_years(text: str | None) -> list[int]:
    years = sorted({int(match.group(0)) for match in YEAR_RE.finditer(text or "")})
    return [year for year in years if 1900 <= year <= 2100]


def _extract_scopes_from_query(
    text: str | None,
    aliases: dict[str, tuple[str, ...]],
) -> frozenset[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return frozenset()
    matched: list[str] = []
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            candidate_normalized = _normalize_text(candidate)
            if not candidate_normalized:
                continue
            if re.search(rf"\b{re.escape(candidate_normalized)}\b", normalized):
                matched.append(canonical)
                break
    return frozenset(matched)


def _extract_country_scopes(query: str) -> frozenset[str]:
    return _extract_scopes_from_query(query, QUERY_COUNTRY_ALIASES)


def _extract_region_scopes(query: str) -> frozenset[str]:
    return _extract_scopes_from_query(query, QUERY_REGION_ALIASES)


def _derive_fallback_regions(country_scopes: frozenset[str]) -> frozenset[str]:
    return frozenset(
        COUNTRY_TO_REGION[country]
        for country in country_scopes
        if country in COUNTRY_TO_REGION
    )


def _is_explanatory_query(query: str) -> bool:
    normalized = " ".join(query.lower().split())
    return normalized.startswith(EXPLANATORY_PREFIXES) or "benefits and challenges" in normalized


def _prefer_recent_documents(query: str, explicit_years: list[int]) -> bool:
    if explicit_years:
        return False
    normalized = " ".join(query.lower().split())
    if any(marker in normalized for marker in ("latest", "current", "recent", "most recent", "as of")):
        return True
    if _is_explanatory_query(normalized):
        return False
    return any(marker in normalized for marker in DATA_MARKERS)


def _classify_query_intent(query: str, explicit_years: list[int]) -> str:
    normalized = " ".join((query or "").lower().split())
    if explicit_years or any(marker in normalized for marker in DATA_MARKERS):
        return "data"
    if any(normalized.startswith(marker) for marker in DEFINITION_MARKERS):
        return "definition"
    if any(marker in normalized for marker in POLICY_MARKERS):
        return "policy"
    if any(marker in normalized for marker in IMPLEMENTATION_MARKERS):
        return "implementation"
    return "concept"


def _build_retrieval_profile(
    query: str, years: int | list[int] | None = None
) -> RetrievalProfile:
    explicit_years: list[int]
    if years is None:
        explicit_years = _extract_years(query)
    elif isinstance(years, int):
        explicit_years = [years]
    else:
        explicit_years = sorted({year for year in years if isinstance(year, int)})
    normalized_query = _normalize_text(query)
    country_scopes = _extract_country_scopes(query)
    region_scopes = _extract_region_scopes(query)
    fallback_region_scopes = _derive_fallback_regions(country_scopes)
    return RetrievalProfile(
        query=query,
        normalized_query=normalized_query,
        query_tokens=_tokenize_text(query),
        query_phrases=_extract_query_phrases(query),
        explicit_years=explicit_years,
        prefer_recent=_prefer_recent_documents(query, explicit_years),
        explanatory=_is_explanatory_query(query),
        intent=_classify_query_intent(query, explicit_years),
        country_scopes=country_scopes,
        region_scopes=region_scopes,
        fallback_region_scopes=fallback_region_scopes,
        has_geographic_scope=bool(country_scopes or region_scopes),
    )


def _trim_query_for_retrieval(query: str) -> str:
    normalized = " ".join((query or "").strip().split())
    lowered = normalized.lower()
    for prefix in sorted(EXPLANATORY_PREFIXES, key=len, reverse=True):
        if lowered.startswith(prefix + " "):
            return normalized[len(prefix) :].strip(" ?.:;,-")
    relationship_prefixes = (
        "what is the connection between ",
        "what is the relation between ",
        "what is the relationship between ",
        "what is the link between ",
        "what is the connection of ",
        "what is the relation of ",
        "what is the relationship of ",
        "what is the link of ",
    )
    for prefix in relationship_prefixes:
        if lowered.startswith(prefix):
            return normalized[len(prefix) :].strip(" ?.:;,-")
    return normalized.strip(" ?.:;,-")


def build_retrieval_queries(query: str) -> list[str]:
    """
    Build retrieval-focused query variants from a raw user question.
    """
    base = " ".join((query or "").strip().split())
    trimmed = _trim_query_for_retrieval(base)
    token_focus = " ".join(
        token
        for token in TOKEN_RE.findall(trimmed.lower())
        if token not in STOPWORDS and token not in RELATIONSHIP_MARKERS
    )
    candidates: list[str] = []
    for candidate in (
        base,
        trimmed,
        token_focus,
        trimmed.replace("feed in tariff", "feed-in tariff"),
        trimmed.replace("feed in tariffs", "feed-in tariffs"),
        (
            trimmed.replace("feed in tariff", "feed-in tariff") + " fit"
            if "feed in tariff" in trimmed
            else ""
        ),
    ):
        candidate = " ".join((candidate or "").split()).strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _prioritize_retrieval_queries(query: str) -> list[str]:
    """
    Prioritize a small number of retrieval query variants to keep lookup bounded.
    """
    base = " ".join((query or "").strip().split())
    trimmed = _trim_query_for_retrieval(base)
    candidates = build_retrieval_queries(query)
    prioritized: list[str] = []

    for candidate in (
        trimmed,
        trimmed.replace("feed in tariff", "feed-in tariff"),
        base,
    ):
        candidate = " ".join((candidate or "").split()).strip()
        if candidate and candidate in candidates and candidate not in prioritized:
            prioritized.append(candidate)

    for candidate in candidates:
        if candidate not in prioritized:
            prioritized.append(candidate)

    normalized_base = _normalize_text(base)
    if any(marker in normalized_base for marker in RELATIONSHIP_MARKERS):
        return prioritized[:2]
    return prioritized[:2]


def _sql_quote(value: str) -> str:
    return value.replace("'", "''")


def _phrase_to_regex(phrase: str) -> str:
    tokens = [token for token in TOKEN_RE.findall((phrase or "").lower()) if len(token) > 1]
    if not tokens:
        return ""
    escaped = [re.escape(token) for token in tokens]
    return r"\b" + r"[-\s]+".join(escaped) + r"\b"


def _build_metadata_patterns(query: str) -> list[str]:
    """
    Build a small set of lexical patterns for title/summary retrieval.

    Prefer multi-word phrases because they are materially more discriminative
    than single-token matches for this corpus.
    """
    trimmed = _trim_query_for_retrieval(query)
    normalized = " ".join((trimmed or "").lower().split())
    raw_tokens = _ordered_raw_tokens(trimmed)
    filtered_tokens = _ordered_query_tokens(trimmed)

    patterns: list[str] = []

    def add_pattern(candidate: str) -> None:
        regex = _phrase_to_regex(candidate)
        if regex and regex not in patterns:
            patterns.append(regex)

    if "feed in tariff" in normalized or "feed-in tariff" in normalized:
        special = r"\bfeed[-\s]+in[-\s]+tariffs?\b"
        patterns.append(special)

    add_pattern(normalized)

    for token_list in (raw_tokens, filtered_tokens):
        max_size = min(4, len(token_list))
        for size in range(max_size, 1, -1):
            for start in range(0, len(token_list) - size + 1):
                add_pattern(" ".join(token_list[start : start + size]))

    if not patterns and filtered_tokens:
        for token in filtered_tokens[:3]:
            add_pattern(token)

    return patterns[:8]


def _merge_candidate_rows(
    existing_rows: list[dict],
    new_rows: list[dict],
) -> list[dict]:
    merged = list(existing_rows)
    seen = {
        (
            (row.get("url") or "").strip(),
            (row.get("title") or "").strip(),
            (row.get("content") or "").strip()[:160],
        )
        for row in existing_rows
    }
    for row in new_rows:
        key = (
            (row.get("url") or "").strip(),
            (row.get("title") or "").strip(),
            (row.get("content") or "").strip()[:160],
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
    return merged


def _safe_list_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value if isinstance(item, str))
    if isinstance(value, str):
        return value
    return ""


def _normalize_scope_values(values: object) -> set[str]:
    normalized: set[str] = set()
    if isinstance(values, list):
        for item in values:
            if isinstance(item, str) and item.strip():
                normalized.add(item.strip().lower())
    elif isinstance(values, str):
        for item in re.split(r"[|,;/]", values):
            item = item.strip().lower()
            if item:
                normalized.add(item)
    return normalized


def _row_country_scopes(row: dict) -> set[str]:
    countries = {
        value.upper() if len(value) == 3 else value.upper()
        for value in _normalize_scope_values(row.get("country_codes"))
    }
    if countries:
        return countries
    geography_text = " ".join(
        filter(
            None,
            [
                _safe_list_text(row.get("country_codes")),
                row.get("geography_tags_text") or "",
                row.get("summary") or "",
                row.get("chunk_summary") or "",
                row.get("content") or "",
                row.get("title") or row.get("canonical_title") or "",
            ],
        )
    )
    return set(_extract_country_scopes(geography_text))


def _row_region_scopes(row: dict) -> set[str]:
    regions = _normalize_scope_values(row.get("region_codes"))
    if regions:
        return regions
    geography_text = " ".join(
        filter(
            None,
            [
                _safe_list_text(row.get("region_codes")),
                row.get("geography_tags_text") or "",
                row.get("summary") or "",
                row.get("chunk_summary") or "",
                row.get("content") or "",
                row.get("title") or row.get("canonical_title") or "",
            ],
        )
    )
    return set(_extract_region_scopes(geography_text))


def _classify_geography_match(row: dict, profile: RetrievalProfile) -> dict[str, object]:
    if not profile.has_geographic_scope:
        return {
            "match_class": "unscoped",
            "rejected": False,
            "country_codes": sorted(_row_country_scopes(row)),
            "region_codes": sorted(_row_region_scopes(row)),
        }

    row_countries = _row_country_scopes(row)
    row_regions = _row_region_scopes(row)
    row_is_global = "global" in row_regions

    if profile.country_scopes:
        if row_countries & profile.country_scopes:
            match_class = "exact_country"
        elif row_countries and not (row_countries & profile.country_scopes):
            match_class = "out_of_scope"
        elif row_regions & profile.fallback_region_scopes:
            match_class = "parent_region"
        elif row_is_global:
            match_class = "global"
        else:
            match_class = "out_of_scope"
    else:
        matching_regions = row_regions & profile.region_scopes
        if matching_regions:
            match_class = "exact_region"
        elif row_countries and any(
            COUNTRY_TO_REGION.get(country) in profile.region_scopes
            for country in row_countries
        ):
            match_class = "exact_region"
        elif row_countries:
            match_class = "out_of_scope"
        elif row_is_global:
            match_class = "global"
        else:
            match_class = "out_of_scope"

    return {
        "match_class": match_class,
        "rejected": match_class == "out_of_scope",
        "country_codes": sorted(row_countries),
        "region_codes": sorted(row_regions),
    }


def _geography_bonus(match_class: str) -> float:
    match match_class:
        case "unscoped":
            return 0.0
        case "exact_country":
            return 2.2
        case "exact_region":
            return 1.7
        case "parent_region":
            return 0.85
        case "global":
            return 0.35
        case "out_of_scope":
            return -4.0
        case _:
            return 0.0


def _allowed_geography_classes(rows: list[dict], profile: RetrievalProfile) -> tuple[set[str] | None, str]:
    if not profile.has_geographic_scope:
        return None, "unscoped"

    classes = {
        str(_classify_geography_match(row, profile)["match_class"])
        for row in rows
    }

    if profile.country_scopes:
        if "exact_country" in classes:
            return {"exact_country", "global"}, "exact_country"
        if "parent_region" in classes:
            return {"parent_region", "global"}, "parent_region"
        if "global" in classes:
            return {"global"}, "global"
        return set(), "out_of_scope"

    if "exact_region" in classes:
        return {"exact_region", "global"}, "exact_region"
    if "global" in classes:
        return {"global"}, "global"
    return set(), "out_of_scope"


def _filter_rows_by_geography(
    rows: list[dict],
    profile: RetrievalProfile,
) -> tuple[list[dict], str, list[dict]]:
    if not rows:
        return [], "empty", []
    allowed_classes, mode = _allowed_geography_classes(rows, profile)
    if allowed_classes is None:
        return rows, mode, []

    filtered: list[dict] = []
    rejected: list[dict] = []
    for row in rows:
        geo = _classify_geography_match(row, profile)
        row_with_geo = {**row, "_geo_match_class": geo["match_class"]}
        if geo["match_class"] in allowed_classes:
            filtered.append(row_with_geo)
        else:
            rejected.append(
                {
                    "title": row.get("canonical_title") or row.get("title"),
                    "document_id": row.get("document_id"),
                    "match_class": geo["match_class"],
                    "country_codes": geo["country_codes"],
                    "region_codes": geo["region_codes"],
                }
            )
    return filtered, mode, rejected


def _authority_tier_bonus(tier: str | None, profile: RetrievalProfile) -> float:
    match (tier or "").lower():
        case "trusted":
            return 1.0 if profile.intent == "data" else 0.6
        case "partner":
            return 0.7 if profile.intent in {"data", "policy"} else 0.45
        case "external":
            return 0.15
        case _:
            return 0.0


def _document_row_breakdown(row: dict, profile: RetrievalProfile) -> dict[str, object]:
    title = row.get("canonical_title") or row.get("title") or ""
    subtitle = row.get("subtitle") or ""
    summary = row.get("summary") or ""
    series_name = row.get("series_name") or ""
    publisher = row.get("publisher") or ""
    document_type = row.get("document_type") or "publication"
    year = row.get("year")
    text_blob = " ".join(filter(None, [title, subtitle, summary, series_name, publisher]))
    topics_text = _safe_list_text(row.get("topic_tags")) or (row.get("topic_tags_text") or "")
    normalized_title = _normalize_text(title)
    normalized_summary = _normalize_text(summary)
    normalized_text_blob = _normalize_text(text_blob)
    normalized_topics_text = _normalize_text(topics_text)
    focus_phrases = _extract_focus_phrases(profile.query)

    title_overlap = _term_overlap_score(profile.query_tokens, title)
    summary_overlap = _term_overlap_score(profile.query_tokens, summary)
    text_overlap = _term_overlap_score(profile.query_tokens, text_blob)
    topics_overlap = _term_overlap_score(profile.query_tokens, topics_text)
    phrase_bonus = _phrase_hits_bonus(profile.query_phrases, text_blob, 0.2)
    title_component = 2.8 * title_overlap
    summary_component = 1.9 * summary_overlap
    text_component = 1.2 * text_overlap
    topics_component = 0.35 * topics_overlap
    exact_title_component = _phrase_match_bonus(profile.normalized_query, title, 1.6)
    exact_summary_component = _phrase_match_bonus(profile.normalized_query, summary, 0.9)
    title_focus_hits = sum(1 for phrase in focus_phrases if phrase in normalized_title)
    summary_focus_hits = sum(1 for phrase in focus_phrases if phrase in normalized_summary)
    text_focus_hits = sum(1 for phrase in focus_phrases if phrase in normalized_text_blob)
    topics_focus_hits = sum(1 for phrase in focus_phrases if phrase in normalized_topics_text)
    focus_hits = max(title_focus_hits, summary_focus_hits, text_focus_hits, topics_focus_hits)
    focus_phrase_component = 0.9 * focus_hits
    recentness_component = _recentness_bonus(year if isinstance(year, int) else None, profile)
    authority_component = _authority_tier_bonus(row.get("authority_tier"), profile)
    source_priority_component = float(row.get("source_priority") or 0.0) * 0.35
    quality_component = float(row.get("quality_score") or 0.0) * 0.8
    flagship_component = 0.0
    if row.get("is_flagship") and profile.intent == "data":
        flagship_component = 0.9
    elif row.get("is_flagship"):
        flagship_component = -0.15
    data_report_component = 0.8 if row.get("is_data_report") and profile.intent == "data" else 0.0
    status_penalty = 3.5 if (row.get("status") or "approved") != "approved" else 0.0
    document_type_component = _document_type_bonus(document_type, profile)
    concept_flagship_penalty = 0.0
    if row.get("is_data_report") and profile.intent == "data":
        data_report_component = 0.8
    if profile.intent in {"concept", "policy"} and document_type == "flagship_report" and summary_overlap < 0.2:
        concept_flagship_penalty = 0.8
    token_hits = len(profile.query_tokens & _tokenize_text(text_blob))
    signal_strength = max(
        title_overlap,
        summary_overlap,
        text_overlap,
        topics_overlap * 0.5,
        0.25 if exact_title_component > 0 else 0.0,
        0.18 if exact_summary_component > 0 else 0.0,
        phrase_bonus / 1.2 if phrase_bonus > 0 else 0.0,
        min(token_hits / max(len(profile.query_tokens) or 1, 1), 1.0),
    )
    signal_component = 1.4 * signal_strength
    no_signal_penalty = 0.0
    if signal_strength < 0.12:
        no_signal_penalty = 4.0
    elif signal_strength < 0.2:
        no_signal_penalty = 1.7
    missing_focus_penalty = 0.0
    if focus_phrases and profile.intent in {"definition", "policy", "implementation"}:
        if focus_hits == 0:
            missing_focus_penalty = 2.8
        elif len(focus_phrases) >= 2 and focus_hits < 2:
            missing_focus_penalty = 1.0
    geo = _classify_geography_match(row, profile)
    geography_component = _geography_bonus(str(geo["match_class"]))
    score = (
        title_component
        + summary_component
        + text_component
        + topics_component
        + phrase_bonus
        + signal_component
        + focus_phrase_component
        + exact_title_component
        + exact_summary_component
        + recentness_component
        + authority_component
        + source_priority_component
        + quality_component
        + flagship_component
        + data_report_component
        + document_type_component
        + geography_component
        - status_penalty
        - concept_flagship_penalty
        - no_signal_penalty
        - missing_focus_penalty
    )
    return {
        "document_id": row.get("document_id"),
        "title": title,
        "source_id": row.get("source_id"),
        "document_type": document_type,
        "year": year,
        "status": row.get("status"),
        "title_overlap": _round_debug(title_overlap),
        "summary_overlap": _round_debug(summary_overlap),
        "text_overlap": _round_debug(text_overlap),
        "topics_overlap": _round_debug(topics_overlap),
        "topics_component": _round_debug(topics_component),
        "phrase_bonus": _round_debug(phrase_bonus),
        "focus_phrases": list(focus_phrases),
        "focus_hits": focus_hits,
        "topics_focus_hits": topics_focus_hits,
        "geography_match_class": geo["match_class"],
        "geography_rejected": geo["rejected"],
        "country_codes": geo["country_codes"],
        "region_codes": geo["region_codes"],
        "geography_component": _round_debug(geography_component),
        "focus_phrase_component": _round_debug(focus_phrase_component),
        "missing_focus_penalty": _round_debug(missing_focus_penalty),
        "signal_strength": _round_debug(signal_strength),
        "signal_component": _round_debug(signal_component),
        "no_signal_penalty": _round_debug(no_signal_penalty),
        "token_hits": token_hits,
        "title_component": _round_debug(title_component),
        "summary_component": _round_debug(summary_component),
        "text_component": _round_debug(text_component),
        "exact_title_component": _round_debug(exact_title_component),
        "exact_summary_component": _round_debug(exact_summary_component),
        "recentness_component": _round_debug(recentness_component),
        "authority_component": _round_debug(authority_component),
        "source_priority_component": _round_debug(source_priority_component),
        "quality_component": _round_debug(quality_component),
        "flagship_component": _round_debug(flagship_component),
        "data_report_component": _round_debug(data_report_component),
        "document_type_component": _round_debug(document_type_component),
        "status_penalty": _round_debug(status_penalty),
        "concept_flagship_penalty": _round_debug(concept_flagship_penalty),
        "score": _round_debug(score),
    }


def _score_document_row(row: dict, profile: RetrievalProfile) -> float:
    return float(_document_row_breakdown(row, profile)["score"])


def _document_has_min_signal(row: dict, profile: RetrievalProfile) -> bool:
    breakdown = _document_row_breakdown(row, profile)
    if bool(breakdown["geography_rejected"]):
        return False
    signal_strength = float(breakdown["signal_strength"])
    title_overlap = float(breakdown["title_overlap"])
    summary_overlap = float(breakdown["summary_overlap"])
    topics_overlap = float(breakdown["topics_overlap"])
    token_hits = int(breakdown["token_hits"])
    focus_hits = int(breakdown["focus_hits"])
    focus_phrases = list(breakdown["focus_phrases"])
    if focus_phrases and focus_hits == 0 and profile.intent in {"definition", "policy", "implementation"}:
        return False
    return (
        signal_strength >= 0.2
        or title_overlap >= 0.14
        or summary_overlap >= 0.1
        or token_hits >= 2
        or topics_overlap >= 0.34
    )


def _select_chunk_rows(
    rows: list[dict],
    profile: RetrievalProfile,
    *,
    limit: int,
) -> list[Chunk]:
    filtered_rows, _mode, _rejected = _filter_rows_by_geography(rows, profile)
    if not filtered_rows:
        return []
    ranked_rows = sorted(
        ((row, _score_chunk_row(row, profile)) for row in filtered_rows),
        key=lambda item: item[1],
        reverse=True,
    )
    chunks: list[Chunk] = []
    for row, _score in ranked_rows[:limit]:
        chunks.append(
            Chunk(
                document_id=row.get("document_id"),
                source=row.get("source_id") or row.get("source"),
                publisher=row.get("publisher"),
                title=row.get("title") or row.get("canonical_title") or "",
                year=int(row.get("year") or 0),
                language=row.get("language") or "",
                url=row.get("url") or "",
                summary=row.get("summary"),
                document_type=row.get("document_type"),
                publication_date=row.get("publication_date"),
                series_name=row.get("series_name"),
                topics=row.get("topic_tags"),
                geographies=_row_geographies(row),
                content=row.get("content") or "",
                chunk_id=row.get("chunk_id"),
                chunk_index=row.get("chunk_index"),
                content_type=row.get("content_type"),
                section_title=row.get("section_title"),
                page_start=row.get("page_start"),
                page_end=row.get("page_end"),
                token_count=row.get("token_count"),
                chunk_summary=row.get("chunk_summary"),
            )
        )
    return chunks


def _build_summary_fallback_chunks(
    documents: list[dict],
    profile: RetrievalProfile,
    *,
    limit: int,
) -> list[Chunk]:
    fallback_rows: list[dict] = []
    filtered_documents, _mode, _rejected = _filter_rows_by_geography(documents, profile)
    for index, row in enumerate(filtered_documents):
        summary = (row.get("summary") or "").strip()
        if not summary:
            continue
        title = row.get("canonical_title") or row.get("title") or ""
        fallback_rows.append(
            {
                **row,
                "title": title,
                "content": summary,
                "chunk_id": row.get("document_id") or f"summary-{index}",
                "chunk_index": 0,
                "content_type": "document_summary",
                "section_title": "Summary",
                "chunk_summary": summary,
                "_distance": 0.45,
            }
        )
    return _select_chunk_rows(fallback_rows, profile, limit=limit)


def _term_overlap_score(query_tokens: set[str], text: str | None) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = _tokenize_text(text)
    if not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def _phrase_match_bonus(normalized_query: str, text: str | None, weight: float) -> float:
    if not normalized_query:
        return 0.0
    normalized_text = _normalize_text(text)
    if normalized_query and normalized_query in normalized_text:
        return weight
    return 0.0


def _phrase_hits_bonus(
    query_phrases: tuple[str, ...], text: str | None, weight: float
) -> float:
    normalized_text = _normalize_text(text)
    if not normalized_text or not query_phrases:
        return 0.0
    bonus = 0.0
    hits = 0
    for phrase in query_phrases:
        if phrase and phrase in normalized_text:
            hits += 1
            bonus += weight * min(len(phrase.split()), 4)
            if hits >= 3:
                break
    return bonus


def _generic_report_penalty(
    title: str,
    profile: RetrievalProfile,
    title_overlap: float,
    summary_overlap: float,
) -> float:
    normalized_title = _normalize_text(title)
    if not profile.explanatory:
        return 0.0
    if title_overlap >= 0.3 or summary_overlap >= 0.25:
        return 0.0
    if any(pattern in normalized_title for pattern in GENERIC_REPORT_PATTERNS):
        return 0.8
    return 0.0


def _infer_source_domain(url: str | None) -> str:
    parsed = urlparse((url or "").strip())
    return parsed.netloc.lower()


def _infer_source_org(source_domain: str) -> str:
    if "trackingsdg7" in source_domain or "esmap" in source_domain:
        return "tracking_sdg7"
    if "undp.org" in source_domain:
        return "undp"
    if "worldbank.org" in source_domain:
        return "world_bank"
    if "iea.org" in source_domain:
        return "iea"
    return "other"


def _infer_document_type(title: str | None, summary: str | None) -> str:
    normalized = _normalize_text(" ".join(filter(None, [title or "", summary or ""])))
    if any(pattern in normalized for pattern in GLOSSARY_PATTERNS):
        return "glossary"
    if any(pattern in normalized for pattern in POLICY_DOCUMENT_PATTERNS):
        return "policy"
    if any(pattern in normalized for pattern in CASE_STUDY_PATTERNS):
        return "case_study"
    if any(pattern in normalized for pattern in FLAGSHIP_REPORT_PATTERNS):
        return "flagship_report"
    if "report" in normalized:
        return "report"
    return "publication"


def _infer_series_id(
    title: str | None,
    url: str | None,
    source_domain: str,
) -> str:
    normalized_title = _normalize_text(title)
    normalized_title = YEAR_RE.sub("", normalized_title)
    normalized_title = " ".join(token for token in normalized_title.split() if token)
    if any(pattern in normalized_title for pattern in FLAGSHIP_REPORT_PATTERNS):
        for pattern in FLAGSHIP_REPORT_PATTERNS:
            if pattern in normalized_title:
                return f"{source_domain}:{pattern}"
    if normalized_title:
        return f"{source_domain}:{normalized_title}"
    return (url or "").strip() or source_domain or "unknown"


def _authority_bonus(source_org: str, profile: RetrievalProfile) -> float:
    match source_org:
        case "tracking_sdg7":
            return 1.2 if profile.intent == "data" else 0.35
        case "undp":
            return 0.7 if profile.intent in {"concept", "policy", "implementation"} else 0.45
        case "world_bank" | "iea":
            return 0.85 if profile.intent == "data" else 0.5
        case _:
            return 0.15


def _document_type_bonus(document_type: str, profile: RetrievalProfile) -> float:
    match profile.intent:
        case "data":
            if document_type == "flagship_report":
                return 1.1
            if document_type == "report":
                return 0.5
            if document_type == "case_study":
                return -0.35
            if document_type == "glossary":
                return -1.2
        case "policy":
            if document_type == "policy":
                return 1.1
            if document_type in {"report", "case_study"}:
                return 0.25
            if document_type == "flagship_report":
                return -0.4
            if document_type == "glossary":
                return -1.6
        case "implementation":
            if document_type == "case_study":
                return 0.95
            if document_type in {"policy", "report"}:
                return 0.25
            if document_type == "glossary":
                return -1.3
        case "definition":
            if document_type == "glossary":
                return 0.5
            if document_type in {"report", "policy"}:
                return 0.15
        case _:
            if document_type in {"report", "flagship_report", "policy"}:
                return 0.45
            if document_type == "case_study":
                return -0.15
            if document_type == "glossary":
                return -1.8
    return 0.0


def _document_specificity_penalty(
    *,
    document_type: str,
    title: str,
    topicality: float,
    profile: RetrievalProfile,
) -> float:
    normalized_title = _normalize_text(title)
    penalty = 0.0
    if document_type == "glossary" and profile.intent != "definition":
        penalty += 1.6
    if profile.intent in {"concept", "policy"} and document_type == "case_study" and topicality < 2.2:
        penalty += 0.45
    if any(pattern in normalized_title for pattern in GENERIC_REPORT_PATTERNS) and topicality < 1.8:
        penalty += 0.7
    return penalty


def _recentness_bonus(year: int | None, profile: RetrievalProfile) -> float:
    if year is None:
        return 0.0
    if profile.explicit_years:
        return 0.6 if year in profile.explicit_years else -0.8
    if not profile.prefer_recent:
        return 0.0
    if year >= 2025:
        return 0.5
    if year >= 2024:
        return 0.35
    if year >= 2022:
        return 0.2
    if year >= 2020:
        return 0.05
    return -0.1


def _round_debug(value: float) -> float:
    return round(float(value), 4)


def _chunk_score_breakdown(row: dict, profile: RetrievalProfile) -> dict[str, object]:
    title = row.get("title") or ""
    summary = row.get("summary") or ""
    content = row.get("content") or ""
    year = row.get("year")
    distance = float(row.get("_distance", 1.0) or 1.0)

    title_overlap = _term_overlap_score(profile.query_tokens, title)
    summary_overlap = _term_overlap_score(profile.query_tokens, summary)
    content_overlap = _term_overlap_score(profile.query_tokens, content)
    title_ratio = (
        SequenceMatcher(None, profile.normalized_query, _normalize_text(title)).ratio()
        if profile.normalized_query and title
        else 0.0
    )
    title_phrase_bonus = _phrase_hits_bonus(profile.query_phrases, title, 0.45)
    summary_phrase_bonus = _phrase_hits_bonus(profile.query_phrases, summary, 0.2)
    content_phrase_bonus = _phrase_hits_bonus(profile.query_phrases, content, 0.1)
    topicality = (
        2.2 * title_overlap
        + 1.4 * summary_overlap
        + 0.8 * content_overlap
        + 0.8 * title_ratio
        + title_phrase_bonus
        + summary_phrase_bonus
        + content_phrase_bonus
    )

    distance_component = 1.4 / (1.0 + max(distance, 0.0))
    title_overlap_component = 2.5 * title_overlap
    summary_overlap_component = 1.3 * summary_overlap
    content_overlap_component = 1.0 * content_overlap
    title_ratio_component = 0.8 * title_ratio
    exact_title_component = _phrase_match_bonus(profile.normalized_query, title, 1.4)
    exact_summary_component = _phrase_match_bonus(profile.normalized_query, summary, 0.8)
    exact_content_component = _phrase_match_bonus(profile.normalized_query, content, 0.6)
    recentness_component = _recentness_bonus(year if isinstance(year, int) else None, profile)
    generic_penalty = _generic_report_penalty(title, profile, title_overlap, summary_overlap)
    low_topicality_penalty = 0.0
    if profile.explanatory and topicality < 1.3:
        low_topicality_penalty = 1.1
    elif topicality < 0.75:
        low_topicality_penalty = 0.6
    geo = _classify_geography_match(row, profile)
    geography_component = _geography_bonus(str(geo["match_class"]))
    score = (
        distance_component
        + title_overlap_component
        + summary_overlap_component
        + content_overlap_component
        + title_ratio_component
        + exact_title_component
        + exact_summary_component
        + exact_content_component
        + title_phrase_bonus
        + summary_phrase_bonus
        + content_phrase_bonus
        + recentness_component
        + geography_component
        - generic_penalty
        - low_topicality_penalty
    )
    return {
        "title": title,
        "year": year,
        "distance": _round_debug(distance),
        "topicality": _round_debug(topicality),
        "title_overlap": _round_debug(title_overlap),
        "summary_overlap": _round_debug(summary_overlap),
        "content_overlap": _round_debug(content_overlap),
        "title_ratio": _round_debug(title_ratio),
        "distance_component": _round_debug(distance_component),
        "title_overlap_component": _round_debug(title_overlap_component),
        "summary_overlap_component": _round_debug(summary_overlap_component),
        "content_overlap_component": _round_debug(content_overlap_component),
        "title_ratio_component": _round_debug(title_ratio_component),
        "exact_title_component": _round_debug(exact_title_component),
        "exact_summary_component": _round_debug(exact_summary_component),
        "exact_content_component": _round_debug(exact_content_component),
        "title_phrase_bonus": _round_debug(title_phrase_bonus),
        "summary_phrase_bonus": _round_debug(summary_phrase_bonus),
        "content_phrase_bonus": _round_debug(content_phrase_bonus),
        "geography_match_class": geo["match_class"],
        "geography_rejected": geo["rejected"],
        "country_codes": geo["country_codes"],
        "region_codes": geo["region_codes"],
        "geography_component": _round_debug(geography_component),
        "recentness_component": _round_debug(recentness_component),
        "generic_report_penalty": _round_debug(generic_penalty),
        "low_topicality_penalty": _round_debug(low_topicality_penalty),
        "score": _round_debug(score),
    }


def _score_chunk_row(row: dict, profile: RetrievalProfile) -> float:
    score = float(_chunk_score_breakdown(row, profile)["score"])
    return score


def _document_candidate_breakdown(
    key: str,
    scored_rows: list[tuple[dict, float]],
    profile: RetrievalProfile,
) -> dict[str, object]:
    best_row = scored_rows[0][0]
    top_rows = [row for row, _ in scored_rows[:3]]
    top_scores = [score for _, score in scored_rows[:3]]

    combined_summary = " ".join((row.get("summary") or "").strip() for row in top_rows)
    combined_content = " ".join((row.get("content") or "").strip() for row in top_rows)
    title = best_row.get("title") or ""
    url = best_row.get("url") or ""
    year = best_row.get("year")
    source_domain = _infer_source_domain(url)
    source_org = _infer_source_org(source_domain)
    document_type = _infer_document_type(title, combined_summary)
    series_id = _infer_series_id(title, url, source_domain)

    title_overlap = _term_overlap_score(profile.query_tokens, title)
    summary_overlap = _term_overlap_score(profile.query_tokens, combined_summary)
    content_overlap = _term_overlap_score(profile.query_tokens, combined_content)
    title_phrase_bonus = _phrase_hits_bonus(profile.query_phrases, title, 0.55)
    summary_phrase_bonus = _phrase_hits_bonus(profile.query_phrases, combined_summary, 0.25)
    content_phrase_bonus = _phrase_hits_bonus(profile.query_phrases, combined_content, 0.14)
    topicality = (
        2.3 * title_overlap
        + 1.9 * summary_overlap
        + 1.4 * content_overlap
        + title_phrase_bonus
        + summary_phrase_bonus
        + content_phrase_bonus
    )

    support_score = sum(
        multiplier * score
        for multiplier, score in zip((1.0, 0.5, 0.25), top_scores, strict=False)
    )
    summary_overlap_component = 1.8 * summary_overlap
    content_overlap_component = 1.4 * content_overlap
    title_overlap_component = 0.9 * title_overlap
    phrase_component = title_phrase_bonus + summary_phrase_bonus + content_phrase_bonus
    recentness_component = _recentness_bonus(year if isinstance(year, int) else None, profile)
    authority_component = _authority_bonus(source_org, profile)
    document_type_component = _document_type_bonus(document_type, profile)
    specificity_penalty = _document_specificity_penalty(
        document_type=document_type,
        title=title,
        topicality=topicality,
        profile=profile,
    )
    low_topicality_penalty = 0.0
    if profile.intent in {"concept", "policy"} and topicality < 1.35:
        low_topicality_penalty = 1.15
    elif topicality < 0.9:
        low_topicality_penalty = 0.65
    geo = _classify_geography_match(best_row, profile)
    geography_component = _geography_bonus(str(geo["match_class"]))

    score = (
        support_score
        + summary_overlap_component
        + content_overlap_component
        + title_overlap_component
        + phrase_component
        + recentness_component
        + authority_component
        + document_type_component
        + geography_component
        - specificity_penalty
        - low_topicality_penalty
    )
    return {
        "key": key,
        "title": title,
        "url": url,
        "year": year,
        "source_domain": source_domain,
        "source_org": source_org,
        "document_type": document_type,
        "series_id": series_id,
        "title_overlap": _round_debug(title_overlap),
        "summary_overlap": _round_debug(summary_overlap),
        "content_overlap": _round_debug(content_overlap),
        "title_phrase_bonus": _round_debug(title_phrase_bonus),
        "summary_phrase_bonus": _round_debug(summary_phrase_bonus),
        "content_phrase_bonus": _round_debug(content_phrase_bonus),
        "topicality": _round_debug(topicality),
        "support_score": _round_debug(support_score),
        "summary_overlap_component": _round_debug(summary_overlap_component),
        "content_overlap_component": _round_debug(content_overlap_component),
        "title_overlap_component": _round_debug(title_overlap_component),
        "phrase_component": _round_debug(phrase_component),
        "recentness_component": _round_debug(recentness_component),
        "authority_component": _round_debug(authority_component),
        "document_type_component": _round_debug(document_type_component),
        "geography_match_class": geo["match_class"],
        "country_codes": geo["country_codes"],
        "region_codes": geo["region_codes"],
        "geography_component": _round_debug(geography_component),
        "specificity_penalty": _round_debug(specificity_penalty),
        "low_topicality_penalty": _round_debug(low_topicality_penalty),
        "score": _round_debug(score),
    }


def _row_geographies(row: dict) -> list[str] | None:
    values: list[str] = []
    for item in list(_row_region_scopes(row)) + list(_row_country_scopes(row)):
        if item not in values:
            values.append(item)
    return values or None


def _build_document_candidate(
    key: str,
    scored_rows: list[tuple[dict, float]],
    profile: RetrievalProfile,
) -> DocumentCandidate:
    breakdown = _document_candidate_breakdown(key, scored_rows, profile)

    return DocumentCandidate(
        key=key,
        series_id=str(breakdown["series_id"]),
        source_domain=str(breakdown["source_domain"]),
        source_org=str(breakdown["source_org"]),
        document_type=str(breakdown["document_type"]),
        score=float(breakdown["score"]),
        topicality=float(breakdown["topicality"]),
        rows=tuple(scored_rows),
    )


def _select_documents_and_chunks(
    rows: list[dict],
    profile: RetrievalProfile,
    *,
    limit: int,
    max_documents: int = 4,
) -> tuple[list[Chunk], list[Document]]:
    filtered_rows, _mode, _rejected = _filter_rows_by_geography(rows, profile)
    if not filtered_rows:
        return [], []

    ranked_rows = []
    for row in filtered_rows:
        ranked_rows.append((row, _score_chunk_row(row, profile)))
    ranked_rows.sort(key=lambda item: item[1], reverse=True)

    per_document: dict[str, list[tuple[dict, float]]] = {}
    for row, score in ranked_rows:
        title = (row.get("title") or "").strip()
        url = (row.get("url") or "").strip()
        key = url or title
        per_document.setdefault(key, []).append((row, score))

    ranked_documents: list[DocumentCandidate] = []
    for key, scored_rows in per_document.items():
        ranked_documents.append(_build_document_candidate(key, scored_rows, profile))
    ranked_documents.sort(key=lambda item: item.score, reverse=True)

    selected_documents: list[Document] = []
    selected_keys = set()
    seen_series: set[str] = set()
    top_score = ranked_documents[0].score if ranked_documents else 0.0
    for candidate in ranked_documents:
        if len(selected_keys) >= max_documents:
            break
        if candidate.series_id in seen_series:
            continue
        best_row = candidate.rows[0][0]
        strong_match = (
            candidate.topicality >= 1.5
            or _term_overlap_score(profile.query_tokens, best_row.get("title")) >= 0.2
            or _term_overlap_score(profile.query_tokens, best_row.get("summary")) >= 0.15
        )
        if len(selected_keys) < 2 or strong_match or candidate.score >= top_score - 1.5:
            selected_keys.add(candidate.key)
            seen_series.add(candidate.series_id)

    for candidate in ranked_documents:
        if candidate.key not in selected_keys:
            continue
        row = candidate.rows[0][0]
        title = (row.get("title") or "").strip()
        url = (row.get("url") or "").strip()
        if not title:
            continue
        selected_documents.append(
            Document(
                document_id=row.get("document_id"),
                source=row.get("source_id") or row.get("source"),
                publisher=row.get("publisher"),
                title=title,
                year=int(row.get("year") or 0),
                language=row.get("language") or "",
                url=url,
                summary=row.get("summary"),
                document_type=row.get("document_type"),
                publication_date=row.get("publication_date"),
                series_name=row.get("series_name"),
                topics=row.get("topic_tags"),
                geographies=_row_geographies(row),
            )
        )
        if len(selected_documents) >= max_documents:
            break

    selected_chunks: list[Chunk] = []
    for row, _ in ranked_rows:
        title = (row.get("title") or "").strip()
        url = (row.get("url") or "").strip()
        key = url or title
        if key not in selected_keys:
            continue
        selected_chunks.append(
            Chunk(
                document_id=row.get("document_id"),
                source=row.get("source_id") or row.get("source"),
                publisher=row.get("publisher"),
                title=title,
                year=int(row.get("year") or 0),
                language=row.get("language") or "",
                url=url,
                summary=row.get("summary"),
                document_type=row.get("document_type"),
                publication_date=row.get("publication_date"),
                series_name=row.get("series_name"),
                topics=row.get("topic_tags"),
                geographies=_row_geographies(row),
                content=row.get("content") or "",
                chunk_id=row.get("chunk_id"),
                chunk_index=row.get("chunk_index"),
                content_type=row.get("content_type"),
                section_title=row.get("section_title"),
                page_start=row.get("page_start"),
                page_end=row.get("page_end"),
                token_count=row.get("token_count"),
                chunk_summary=row.get("chunk_summary"),
            )
        )
        if len(selected_chunks) >= limit:
            break

    if not selected_chunks:
        fallback_rows = [row for row, _ in ranked_rows[:limit]]
        selected_chunks = [
            Chunk(
                document_id=row.get("document_id"),
                source=row.get("source_id") or row.get("source"),
                publisher=row.get("publisher"),
                title=row.get("title") or "",
                year=int(row.get("year") or 0),
                language=row.get("language") or "",
                url=row.get("url") or "",
                summary=row.get("summary"),
                document_type=row.get("document_type"),
                publication_date=row.get("publication_date"),
                series_name=row.get("series_name"),
                topics=row.get("topic_tags"),
                geographies=_row_geographies(row),
                content=row.get("content") or "",
                chunk_id=row.get("chunk_id"),
                chunk_index=row.get("chunk_index"),
                content_type=row.get("content_type"),
                section_title=row.get("section_title"),
                page_start=row.get("page_start"),
                page_end=row.get("page_end"),
                token_count=row.get("token_count"),
                chunk_summary=row.get("chunk_summary"),
            )
            for row in fallback_rows
        ]

    return selected_chunks, selected_documents


def get_storage_options() -> dict[str, str]:
    """
    Get storage options for Azure Blob Storage backend.

    The options can be passed to LanceDB or pandas to connect to remote storage.

    Returns
    -------
    dict
        Mapping storage options.
    """
    sas_url = os.getenv("STORAGE_SAS_URL")
    if sas_url:
        parsed = urlparse(sas_url)
        host = parsed.netloc.split(":")[0]
        account_name = host.split(".")[0] if host else ""
        query = parsed.query.lstrip("?")
        if account_name and query:
            return {"account_name": account_name, "sas_token": query}
        raise RuntimeError(
            "Invalid `STORAGE_SAS_URL`. Expected a full Azure Blob SAS URL like "
            "`https://<account>.blob.core.windows.net/...?...`."
        )

    account_name = os.getenv("STORAGE_ACCOUNT_NAME")
    account_key = os.getenv("STORAGE_ACCOUNT_KEY")
    sas_token = os.getenv("STORAGE_SAS_TOKEN")
    if not account_name:
        raise RuntimeError(
            "Missing `STORAGE_ACCOUNT_NAME` environment variable. "
            "Populate it in your `.env` file."
        )
    if account_key:
        return {"account_name": account_name, "account_key": account_key}
    if sas_token:
        return {"account_name": account_name, "sas_token": sas_token}
    raise RuntimeError(
        "Missing storage credentials. Set either `STORAGE_ACCOUNT_KEY` or "
        "`STORAGE_SAS_TOKEN` in your `.env` file."
    )


async def get_connection() -> lancedb.AsyncConnection:
    """
    Get an asynchronous database connection for LanceDB stored on Azure Blob Storage.

    Returns
    -------
    lancedb.AsyncConnection
        Asynchronous database connection client.
    """
    return await lancedb.connect_async(
        "az://lancedb", storage_options=get_storage_options()
    )


class Client:
    """
    Database client class to perform routine operations using LanceDB.
    """

    def __init__(self, connection: lancedb.AsyncConnection):
        self.connection = connection
        self.embedder = genai.get_embedding_client()

    async def table_exists(self, name: str) -> bool:
        table_names = await self.connection.table_names()
        return name in set(table_names)

    async def open_optional_table(self, name: str):
        try:
            return await self.connection.open_table(name)
        except ValueError as error:
            if "was not found" in str(error):
                return None
            raise

    async def get_table_field_names(self, name: str) -> set[str]:
        table = await self.open_optional_table(name)
        if table is None:
            return set()
        schema = await table.schema()
        return {field.name for field in schema}

    async def get_document_index_rows(
        self,
        *,
        refresh: bool = False,
    ) -> list[dict]:
        documents_table = await self.open_optional_table("documents")
        if documents_table is None:
            return []

        ttl_seconds = _env_timeout_seconds(
            "DOCUMENT_INDEX_CACHE_TTL_SECONDS",
            DEFAULT_DOCUMENT_CACHE_TTL_SECONDS,
            min_value=1.0,
        )
        cached_rows = _DOCUMENT_INDEX_CACHE.get("rows")
        cache_age = monotonic() - float(_DOCUMENT_INDEX_CACHE.get("loaded_at") or 0.0)
        connection_id = id(self.connection)
        if (
            not refresh
            and isinstance(cached_rows, list)
            and cached_rows
            and _DOCUMENT_INDEX_CACHE.get("connection_id") == connection_id
            and cache_age <= ttl_seconds
        ):
            return list(cached_rows)

        schema = await documents_table.schema()
        field_names = {field.name for field in schema}
        select_fields = [field for field in DOCUMENT_INDEX_FIELDS if field in field_names]
        rows = await documents_table.query().select(select_fields).to_list()
        approved_rows = [row for row in rows if (row.get("status") or "approved") == "approved"]
        _DOCUMENT_INDEX_CACHE["rows"] = approved_rows
        _DOCUMENT_INDEX_CACHE["loaded_at"] = monotonic()
        _DOCUMENT_INDEX_CACHE["connection_id"] = connection_id
        return list(approved_rows)

    async def _overwrite_table_rows(
        self,
        name: str,
        rows: list[dict],
        *,
        key_field: str,
    ) -> int:
        existing_table = await self.open_optional_table(name)
        merged: dict[str, dict] = {}
        if existing_table is not None:
            existing_rows = await existing_table.query().to_list()
            for row in existing_rows:
                key = str(row.get(key_field) or "")
                if key:
                    merged[key] = row
        for row in rows:
            key = str(row.get(key_field) or "")
            if key:
                merged[key] = row
        ordered_rows = list(merged.values())
        await self.connection.create_table(
            name,
            data=ordered_rows,
            mode="overwrite",
        )
        return len(rows)

    async def bootstrap_corpus_tables(
        self,
        *,
        overwrite: bool = False,
        rewrite_chunks: bool = False,
        parser_version: str = "bootstrap-v1",
        embedding_version: str = "legacy",
        status: str = "approved",
        progress=None,
    ) -> dict[str, int]:
        async def emit(message: str) -> None:
            if progress is None:
                return
            result = progress(message)
            if isawaitable(result):
                await result

        await emit("Opening chunks table...")
        chunks_table = await self.open_optional_table("chunks")
        if chunks_table is None:
            await emit("Chunks table not found. Nothing to bootstrap.")
            return {"sources": 0, "documents": 0, "chunks": 0}

        await emit("Counting rows in chunks table...")
        chunk_count = await chunks_table.count_rows()
        await emit(f"Chunks table rows: {chunk_count}")
        await emit("Loading chunk rows from storage...")
        rows = await chunks_table.query().to_list()
        await emit(f"Loaded {len(rows)} chunk rows.")
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(corpus.document_key_from_row(row), []).append(row)
        await emit(f"Grouped rows into {len(grouped)} candidate documents.")

        documents: list[DocumentRecord] = []
        for key in sorted(grouped):
            documents.append(
                corpus.build_document_record(
                    grouped[key],
                    parser_version=parser_version,
                    embedding_version=embedding_version,
                    status=status,
                )
            )
        await emit(f"Built {len(documents)} document records.")
        sources = corpus.build_source_records_for_documents(documents)
        await emit(f"Built {len(sources)} source records.")
        documents_by_key = {
            key: document
            for key, document in zip(sorted(grouped), documents, strict=False)
        }
        enriched_rows = corpus.enrich_chunk_rows(rows, documents_by_key=documents_by_key)
        await emit(f"Prepared {len(enriched_rows)} enriched chunk rows.")

        table_mode = "overwrite" if overwrite else None
        await emit("Writing sources table...")
        await self.connection.create_table(
            "sources",
            data=[item.model_dump() for item in sources],
            mode=table_mode,
            exist_ok=not overwrite,
        )
        await emit("Writing documents table...")
        await self.connection.create_table(
            "documents",
            data=[item.model_dump() for item in documents],
            mode=table_mode,
            exist_ok=not overwrite,
        )
        if rewrite_chunks:
            await emit("Rewriting chunks table with document provenance...")
            await self.connection.create_table(
                "chunks",
                data=enriched_rows,
                mode="overwrite",
            )
        await emit("Bootstrap completed.")

        return {
            "sources": len(sources),
            "documents": len(documents),
            "chunks": len(enriched_rows) if rewrite_chunks else 0,
        }

    async def upsert_sources(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        return await self._overwrite_table_rows("sources", rows, key_field="source_id")

    async def upsert_documents(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        return await self._overwrite_table_rows("documents", rows, key_field="document_id")

    async def upsert_chunks(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        table = await self.open_optional_table("chunks")
        if table is None:
            await self.connection.create_table("chunks", data=rows)
            return len(rows)
        schema = await table.schema()
        field_names = {field.name for field in schema}
        if rows and "chunk_id" in rows[0]:
            required_fields = {
                "document_id",
                "chunk_id",
                "chunk_index",
                "content_type",
                "section_title",
                "page_start",
                "page_end",
                "token_count",
                "chunk_summary",
            }
            if not required_fields.issubset(field_names):
                missing = sorted(required_fields - field_names)
                raise RuntimeError(
                    "Chunks table is not yet provenance-aware. "
                    "Re-run bootstrap with `--rewrite-chunks` before importing enriched chunks. "
                    f"Missing fields: {', '.join(missing)}"
                )
            return await self._overwrite_table_rows("chunks", rows, key_field="chunk_id")
        await table.add(rows)
        return len(rows)

    async def search_nodes(self, pattern: str = "", limit: int = 10) -> list[Node]:
        """
        Search nodes in the graph, optionally utilising RegEx patterns.

        The search is case insentitive.

        Parameters
        ----------
        pattern : str, optional
            Optional pattern to match the nodes.
        limit : int, default=10
            Number of matching nodes to return.

        Returns
        -------
        list[Node]
            A list of nodes sorted by name.
        """
        try:
            table = await self.connection.open_table("nodes")
        except ValueError as error:
            if "was not found" in str(error):
                return []
            raise
        return sorted(
            [
                Node(**node)
                for node in await table.query()
                .where(f"regexp_like(name, '(?i){pattern}')")
                .limit(limit)
                .to_list()
            ]
        )

    async def find_node(self, query: str, method: SearchMethod) -> Node | None:
        """
        Find the node that best matches the query.

        Parameters
        ----------
        query : str
            Plain text query.
        method : {SearchMethod.EXACT, SearchMethod.VECTOR}
            Search method to utilise.

        Returns
        -------
        Node | None
            A matching node if found, otherwise None is returned.
        """
        try:
            table = await self.connection.open_table("nodes")
        except ValueError as error:
            if "was not found" in str(error):
                return None
            raise
        match method:
            case SearchMethod.VECTOR:
                vector = await self.embedder.aembed_query(query)
                results = table.vector_search(vector)
            case SearchMethod.EXACT:
                # case insensitive
                results = table.query().where(f"lower(name) == '{query.lower()}'")
            case _:
                raise ValueError(f"Method {method} is not supported.")
        if not (nodes := await results.limit(1).to_list()):
            return None
        return Node(**nodes[0])

    async def find_subgraph(
        self, graph: nx.Graph, query: str | list[str], hops: int = 2
    ) -> Graph:
        """
        Find a subgraph relevant to a given query.

        Parameters
        ----------
        query : str
            Plain text user query.
        hops : int, default=2
            The size of the neighbourhood to extract the graph from.

        Returns
        -------
        Graph
            Object with node and edge lists.
        """
        def lexical_source(candidate_query: str) -> str | None:
            norm_query = " ".join(re.findall(r"[a-z0-9]+", candidate_query.lower()))
            if not norm_query:
                return None
            exact = [
                name
                for name in graph.nodes
                if isinstance(name, str)
                and " ".join(re.findall(r"[a-z0-9]+", name.lower())) == norm_query
            ]
            if exact:
                return max(
                    exact,
                    key=lambda name: float(graph.nodes[name].get("weight", 0.0) or 0.0),
                )
            query_tokens = set(norm_query.split())
            if not query_tokens:
                return None
            best_name = None
            best_score = 0.0
            for name in graph.nodes:
                if not isinstance(name, str):
                    continue
                norm_name = " ".join(re.findall(r"[a-z0-9]+", name.lower()))
                if not norm_name:
                    continue
                name_tokens = set(norm_name.split())
                if not name_tokens:
                    continue
                overlap = len(query_tokens & name_tokens) / max(1, len(name_tokens))
                ratio = SequenceMatcher(None, norm_query, norm_name).ratio()
                score = max(overlap, ratio)
                if score > best_score:
                    best_score = score
                    best_name = name
            if best_name and best_score >= 0.45:
                return best_name
            return None

        async def vector_source(candidate_query: str) -> str | None:
            timeout_seconds = _env_timeout_seconds(
                "GRAPH_VECTOR_TIMEOUT_SECONDS",
                20.0,
            )
            try:
                node = await asyncio.wait_for(
                    self.find_node(candidate_query, SearchMethod.VECTOR),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "find_subgraph vector source timeout after %ss for query=%r",
                    timeout_seconds,
                    candidate_query,
                )
                return None
            except Exception as error:
                logger.warning(
                    "find_subgraph vector source failed for query=%r: %s",
                    candidate_query,
                    error,
                )
                return None
            return node.name if node is not None else None

        # match query or queries to find central nodes
        queries = [query] if isinstance(query, str) else query
        sources: list[str] = []
        for candidate_query in queries:
            source = lexical_source(candidate_query)
            if source is None:
                source = await vector_source(candidate_query)
            if source is None and graph.number_of_nodes() > 0:
                source = max(
                    graph.nodes,
                    key=lambda name: float(graph.nodes[name].get("weight", 0.0) or 0.0),
                )
            if source and source in graph and source not in sources:
                sources.append(source)
        if not sources:
            return Graph(nodes=[], edges=[])
        nodes = utils.get_neighbourhood_nodes(graph, sources, hops)
        graph = graph.subgraph(nodes).copy()
        nodes = utils.get_closest_nodes(graph, sources)
        graph = graph.subgraph(nodes).copy()
        graph = utils.prune_edges(graph)
        # pass all sources to compute neighbourhood from nearest central node
        return Graph.from_networkx(graph, sources)

    async def retrieve_chunks(
        self,
        query: str,
        years: int | list[int] | None = None,
        limit: int = 20,
        debug: dict | None = None,
    ) -> tuple[list[Chunk], list[Document]]:
        """
        Retrieve the document chunks from the database that best match a query.

        Parameters
        ----------
        query : str
            Plain text user query.
        years : int | list[int] | None, optional
            Optional explicit year filter extracted from the user query.
        limit : int, default=20
            Maximum number of best matching chunks to retrieve.

        Returns
        -------
        tuple[list[Chunk], list[Document]]
            Curated chunks for answer grounding and a separate ranked document list.
        """
        table = await self.connection.open_table("chunks")
        if hasattr(table, "schema"):
            chunk_schema = await table.schema()
            chunk_fields = {field.name for field in chunk_schema}
        else:
            chunk_fields = {
                "document_id",
                "title",
                "year",
                "language",
                "url",
                "summary",
                "content",
            }
        documents_table = await self.open_optional_table("documents")
        profile = _build_retrieval_profile(query, years)
        candidate_limit = max(limit * 3, 24)
        retrieval_queries = _prioritize_retrieval_queries(query)
        lexical_timeout = _env_timeout_seconds(
            "RETRIEVE_CHUNKS_LEXICAL_TIMEOUT_SECONDS",
            6.0,
        )
        document_index_timeout = _env_timeout_seconds(
            "RETRIEVE_DOCUMENT_INDEX_TIMEOUT_SECONDS",
            max(lexical_timeout, 15.0),
        )
        per_variant_timeout = _env_timeout_seconds(
            "RETRIEVE_CHUNKS_VARIANT_TIMEOUT_SECONDS",
            12.0,
        )
        lexical_rows: list[dict] = []
        vector_rows: list[dict] = []

        logger.info(
            (
                "retrieve_chunks start query=%r variants=%r candidate_limit=%s "
                "lexical_timeout=%ss variant_timeout=%ss"
            ),
            query,
            retrieval_queries,
            candidate_limit,
            lexical_timeout,
            per_variant_timeout,
        )
        if debug is not None:
            debug.update(
                {
                    "query": query,
                    "limit": limit,
                    "candidate_limit": candidate_limit,
                    "profile": {
                        "normalized_query": profile.normalized_query,
                        "query_tokens": sorted(profile.query_tokens),
                        "query_phrases": list(profile.query_phrases),
                        "explicit_years": profile.explicit_years,
                        "prefer_recent": profile.prefer_recent,
                        "explanatory": profile.explanatory,
                        "intent": profile.intent,
                        "country_scopes": sorted(profile.country_scopes),
                        "region_scopes": sorted(profile.region_scopes),
                        "fallback_region_scopes": sorted(profile.fallback_region_scopes),
                        "has_geographic_scope": profile.has_geographic_scope,
                    },
                    "retrieval_queries": retrieval_queries,
                    "documents_table_present": documents_table is not None,
                    "branches": [],
                    "selected": {},
                }
            )

        def apply_year_where(base_where: str | None = None) -> str | None:
            clauses: list[str] = []
            if base_where:
                clauses.append(f"({base_where})")
            if profile.explicit_years:
                if len(profile.explicit_years) == 1:
                    clauses.append(f"(year = {profile.explicit_years[0]})")
                else:
                    joined_years = ", ".join(map(str, profile.explicit_years))
                    clauses.append(f"(year IN ({joined_years}))")
            if not clauses:
                return None
            return " AND ".join(clauses)

        document_index_rows: list[dict] | None = None

        async def fetch_document_index_rows() -> list[dict]:
            nonlocal document_index_rows
            if documents_table is None:
                return []
            if document_index_rows is None:
                document_index_rows = await self.get_document_index_rows()
            return document_index_rows

        async def fetch_lexical_rows(retrieval_query: str) -> list[dict]:
            patterns = _build_metadata_patterns(retrieval_query)
            if not patterns:
                return []
            pattern_clauses = []
            for pattern in patterns:
                escaped = _sql_quote(pattern)
                pattern_clauses.append(f"regexp_like(title, '(?i){escaped}')")
                pattern_clauses.append(f"regexp_like(summary, '(?i){escaped}')")
            where_clause = apply_year_where(" OR ".join(pattern_clauses))
            select_fields = [
                field
                for field in [
                    "document_id",
                    "title",
                    "year",
                    "language",
                    "url",
                    "summary",
                    "content",
                    "region_codes",
                    "country_codes",
                    "geography_tags_text",
                    "topic_tags",
                    "publisher",
                    "document_type",
                    "publication_date",
                    "series_name",
                    "chunk_id",
                    "chunk_index",
                    "content_type",
                    "section_title",
                    "page_start",
                    "page_end",
                    "token_count",
                    "chunk_summary",
                ]
                if field in chunk_fields
            ]
            query_builder = table.query().select(select_fields)
            if where_clause:
                query_builder = query_builder.where(where_clause)
            return await query_builder.limit(candidate_limit).to_list()

        async def fetch_variant_rows(retrieval_query: str) -> list[dict]:
            vector = await self.embedder.aembed_query(retrieval_query)
            search = table.vector_search(vector)
            year_where = apply_year_where()
            if year_where:
                search = search.where(year_where)
            return await search.limit(candidate_limit).to_list()

        async def fetch_document_chunks(
            shortlisted_documents: list[dict],
            retrieval_query: str,
            *,
            lexical_only: bool,
        ) -> list[dict]:
            if not shortlisted_documents:
                return []
            document_ids = [
                row.get("document_id")
                for row in shortlisted_documents
                if isinstance(row.get("document_id"), str) and row.get("document_id")
            ]
            url_values = [
                (row.get("url") or "").strip()
                for row in shortlisted_documents
                if (row.get("url") or "").strip()
            ]
            title_values = [
                (row.get("canonical_title") or row.get("title") or "").strip()
                for row in shortlisted_documents
                if (row.get("canonical_title") or row.get("title") or "").strip()
            ]
            filters = []
            if "document_id" in chunk_fields and document_ids:
                joined = ", ".join(f"'{_sql_quote(value)}'" for value in document_ids)
                filters.append(f"document_id IN ({joined})")
            if "url" in chunk_fields and url_values:
                joined = ", ".join(f"'{_sql_quote(value)}'" for value in url_values)
                filters.append(f"url IN ({joined})")
            if "title" in chunk_fields and title_values:
                joined = ", ".join(f"'{_sql_quote(value)}'" for value in title_values)
                filters.append(f"title IN ({joined})")
            if not filters:
                return []

            base_filter = "(" + " OR ".join(filters) + ")"
            where_clause = apply_year_where(base_filter)
            scoped_limit = max(candidate_limit * 8, 240)

            if lexical_only:
                query_builder = table.query()
                if where_clause:
                    query_builder = query_builder.where(where_clause)
                return await query_builder.limit(scoped_limit).to_list()

            vector = await self.embedder.aembed_query(retrieval_query)
            search = table.vector_search(vector)
            if where_clause:
                search = search.where(where_clause)
            return await search.limit(candidate_limit).to_list()

        if documents_table is not None:
            try:
                index_rows = await asyncio.wait_for(
                    fetch_document_index_rows(),
                    timeout=document_index_timeout,
                )
            except asyncio.TimeoutError:
                index_rows = []
                logger.warning(
                    "retrieve_document_index timeout query=%r timeout=%ss",
                    query,
                    document_index_timeout,
                )
                if debug is not None:
                    debug["branches"].append(
                        {
                            "stage": "document_index_load",
                            "status": "timeout",
                            "timeout_seconds": document_index_timeout,
                        }
                    )
            except Exception as error:
                index_rows = []
                logger.warning(
                    "retrieve_document_index failed query=%r error=%s",
                    query,
                    error,
                )
                if debug is not None:
                    debug["branches"].append(
                        {
                            "stage": "document_index_load",
                            "status": "error",
                            "error": str(error),
                        }
                    )
            else:
                logger.info(
                    "retrieve_document_index loaded query=%r rows=%s",
                    query,
                    len(index_rows),
                )
                if debug is not None:
                    debug["branches"].append(
                        {
                            "stage": "document_index_load",
                            "rows": len(index_rows),
                            "source": "cache",
                        }
                    )

            ranked_documents: list[DocumentHit] = []
            ranked_document_debug: list[dict] = []
            for retrieval_query in retrieval_queries:
                variant_profile = _build_retrieval_profile(retrieval_query, profile.explicit_years)
                filtered_index_rows, geography_mode, rejected_rows = _filter_rows_by_geography(
                    index_rows,
                    variant_profile,
                )
                variant_hits: list[tuple[DocumentHit, dict]] = []
                for row in filtered_index_rows:
                    year = row.get("year")
                    if profile.explicit_years and year not in profile.explicit_years:
                        continue
                    breakdown = _document_row_breakdown(row, variant_profile)
                    if not _document_has_min_signal(row, variant_profile):
                        continue
                    score = float(breakdown["score"])
                    if score < 0.8:
                        continue
                    variant_hits.append((DocumentHit(row=row, score=score), breakdown))

                variant_hits.sort(key=lambda item: item[0].score, reverse=True)
                if debug is not None:
                    debug["branches"].append(
                        {
                            "stage": "document_candidates_index",
                            "variant": retrieval_query,
                            "geography_mode": geography_mode,
                            "rows": len(variant_hits),
                            "rejected_out_of_scope": rejected_rows[:10],
                            "sample_titles": [
                                item.row.get("canonical_title") or item.row.get("title")
                                for item, _ in variant_hits[:5]
                            ],
                        }
                    )
                logger.info(
                    "retrieve_document_candidates index query=%r variant=%r rows=%s",
                    query,
                    retrieval_query,
                    len(variant_hits),
                )
                if variant_profile.has_geographic_scope and not variant_hits:
                    logger.info(
                        "retrieve_document_candidates geography filtered query=%r variant=%r mode=%s rejected=%s",
                        query,
                        retrieval_query,
                        geography_mode,
                        len(rejected_rows),
                    )
                for hit, breakdown in variant_hits:
                    key = str(
                        hit.row.get("document_id")
                        or hit.row.get("url")
                        or hit.row.get("canonical_title")
                        or hit.row.get("title")
                        or ""
                    )
                    existing = next(
                        (
                            item
                            for item in ranked_documents
                            if str(
                                item.row.get("document_id")
                                or item.row.get("url")
                                or item.row.get("canonical_title")
                                or item.row.get("title")
                                or ""
                            )
                            == key
                        ),
                        None,
                    )
                    if existing is None or hit.score > existing.score:
                        if existing is not None:
                            ranked_documents.remove(existing)
                        ranked_documents.append(hit)
                    if len(ranked_document_debug) < 20:
                        ranked_document_debug.append(
                            {
                                **breakdown,
                                "variant": retrieval_query,
                                "topic_tags": hit.row.get("topic_tags"),
                                "country_codes": hit.row.get("country_codes"),
                                "region_codes": hit.row.get("region_codes"),
                                "geography_mode": geography_mode,
                            }
                        )

            ranked_documents.sort(key=lambda item: item.score, reverse=True)

            if ranked_documents:
                shortlisted_rows = [item.row for item in ranked_documents[:6]]
                api_documents = [corpus.document_record_to_api(row) for row in shortlisted_rows]
                if debug is not None:
                    debug["selected"]["document_candidates"] = ranked_document_debug[:10]
                chunk_rows: list[dict] = []

                for retrieval_query in retrieval_queries:
                    try:
                        rows = await asyncio.wait_for(
                            fetch_document_chunks(shortlisted_rows, retrieval_query, lexical_only=True),
                            timeout=lexical_timeout,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "retrieve_document_chunks lexical timeout query=%r variant=%r timeout=%ss",
                            query,
                            retrieval_query,
                            lexical_timeout,
                        )
                        if debug is not None:
                            debug["branches"].append(
                                {
                                    "stage": "document_chunks_lexical",
                                    "variant": retrieval_query,
                                    "status": "timeout",
                                    "patterns": _build_metadata_patterns(retrieval_query),
                                }
                            )
                        continue
                    except Exception as error:
                        logger.warning(
                            "retrieve_document_chunks lexical failed query=%r variant=%r error=%s",
                            query,
                            retrieval_query,
                            error,
                        )
                        if debug is not None:
                            debug["branches"].append(
                                {
                                    "stage": "document_chunks_lexical",
                                    "variant": retrieval_query,
                                    "status": "error",
                                    "error": str(error),
                                    "patterns": _build_metadata_patterns(retrieval_query),
                                }
                            )
                        continue
                    for row in rows:
                        matching = next(
                            (
                                document
                                for document in shortlisted_rows
                                if (
                                    row.get("document_id")
                                    and row.get("document_id") == document.get("document_id")
                                )
                                or (
                                    (row.get("url") or "").strip()
                                    and (row.get("url") or "").strip() == (document.get("url") or "").strip()
                                )
                                or (
                                    (row.get("title") or "").strip()
                                    and (row.get("title") or "").strip()
                                    == (document.get("canonical_title") or document.get("title") or "").strip()
                                )
                            ),
                            None,
                        )
                        if matching:
                            row = {**matching, **row}
                        chunk_rows = _merge_candidate_rows(chunk_rows, [row])
                    if debug is not None:
                        debug["branches"].append(
                            {
                                "stage": "document_chunks_lexical",
                                "variant": retrieval_query,
                                "rows": len(rows),
                                "patterns": _build_metadata_patterns(retrieval_query),
                                "sample_titles": [
                                    row.get("title") or row.get("canonical_title")
                                    for row in rows[:5]
                                ],
                            }
                        )

                if not chunk_rows:
                    for retrieval_query in retrieval_queries:
                        try:
                            rows = await asyncio.wait_for(
                                fetch_document_chunks(shortlisted_rows, retrieval_query, lexical_only=False),
                                timeout=per_variant_timeout,
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "retrieve_document_chunks vector timeout query=%r variant=%r timeout=%ss",
                                query,
                                retrieval_query,
                                per_variant_timeout,
                            )
                            if debug is not None:
                                debug["branches"].append(
                                    {
                                        "stage": "document_chunks_vector",
                                        "variant": retrieval_query,
                                        "status": "timeout",
                                        "patterns": _build_metadata_patterns(retrieval_query),
                                    }
                                )
                            continue
                        except Exception as error:
                            logger.warning(
                                "retrieve_document_chunks vector failed query=%r variant=%r error=%s",
                                query,
                                retrieval_query,
                                error,
                            )
                            if debug is not None:
                                debug["branches"].append(
                                    {
                                        "stage": "document_chunks_vector",
                                        "variant": retrieval_query,
                                        "status": "error",
                                        "error": str(error),
                                        "patterns": _build_metadata_patterns(retrieval_query),
                                    }
                                )
                            continue
                        for row in rows:
                            matching = next(
                                (
                                    document
                                    for document in shortlisted_rows
                                    if (
                                        row.get("document_id")
                                        and row.get("document_id") == document.get("document_id")
                                    )
                                    or (
                                        (row.get("url") or "").strip()
                                        and (row.get("url") or "").strip() == (document.get("url") or "").strip()
                                    )
                                    or (
                                        (row.get("title") or "").strip()
                                        and (row.get("title") or "").strip()
                                        == (document.get("canonical_title") or document.get("title") or "").strip()
                                    )
                                ),
                                None,
                            )
                            if matching:
                                row = {**matching, **row}
                            chunk_rows = _merge_candidate_rows(chunk_rows, [row])
                        if debug is not None:
                            debug["branches"].append(
                                {
                                    "stage": "document_chunks_vector",
                                    "variant": retrieval_query,
                                    "rows": len(rows),
                                    "patterns": _build_metadata_patterns(retrieval_query),
                                    "sample_titles": [
                                        row.get("title") or row.get("canonical_title")
                                        for row in rows[:5]
                                    ],
                                }
                            )

                chunk_rows, chunk_geography_mode, rejected_chunk_rows = _filter_rows_by_geography(
                    chunk_rows,
                    profile,
                )
                selected_chunks = _select_chunk_rows(chunk_rows, profile, limit=limit)
                if not selected_chunks and api_documents:
                    selected_chunks = _build_summary_fallback_chunks(
                        shortlisted_rows,
                        profile,
                        limit=min(limit, len(shortlisted_rows)),
                    )
                logger.info(
                    "retrieve_chunks document-first selection query=%r docs=%s chunks=%s",
                    query,
                    len(api_documents),
                    len(selected_chunks),
                )
                if debug is not None:
                    debug["selected"]["documents"] = [
                        document.model_dump()
                        for document in api_documents
                    ]
                    debug["selected"]["chunks"] = [
                        {
                            "document_id": chunk.document_id,
                            "chunk_id": chunk.chunk_id,
                            "title": chunk.title,
                            "section_title": chunk.section_title,
                            "year": chunk.year,
                            "url": chunk.url,
                        }
                        for chunk in selected_chunks
                    ]
                    debug["selected"]["path"] = "document_first"
                    debug["selected"]["geography_mode"] = chunk_geography_mode
                    debug["selected"]["rejected_out_of_scope"] = rejected_chunk_rows[:10]
                    debug["selected"]["chunk_candidates"] = [
                        _chunk_score_breakdown(row, profile)
                        for row in chunk_rows[:10]
                    ]
                if selected_chunks or api_documents:
                    return selected_chunks, api_documents
                logger.warning(
                    "retrieve_chunks document-first produced no usable output query=%r top_candidates=%s",
                    query,
                    [
                        {
                            "title": item.row.get("canonical_title") or item.row.get("title"),
                            "score": round(item.score, 4),
                            "document_type": item.row.get("document_type"),
                            "status": item.row.get("status"),
                        }
                        for item in ranked_documents[:5]
                    ],
                )
            else:
                logger.warning(
                    "retrieve_chunks documents table present but no lexical document candidates query=%r variants=%r",
                    query,
                    retrieval_queries,
                )
                if debug is not None:
                    debug["selected"]["path"] = "document_first_no_candidates"

        for variant_index, retrieval_query in enumerate(retrieval_queries):
            try:
                variant_rows = await asyncio.wait_for(
                    fetch_lexical_rows(retrieval_query),
                    timeout=lexical_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "retrieve_chunks lexical timeout query=%r variant=%r timeout=%ss",
                    query,
                    retrieval_query,
                    lexical_timeout,
                )
                if debug is not None:
                    debug["branches"].append(
                        {
                            "stage": "chunk_candidates_lexical",
                            "variant": retrieval_query,
                            "status": "timeout",
                            "patterns": _build_metadata_patterns(retrieval_query),
                        }
                    )
                continue
            except Exception as error:
                logger.warning(
                    "retrieve_chunks lexical failed query=%r variant=%r error=%s",
                    query,
                    retrieval_query,
                    error,
                )
                if debug is not None:
                    debug["branches"].append(
                        {
                            "stage": "chunk_candidates_lexical",
                            "variant": retrieval_query,
                            "status": "error",
                            "error": str(error),
                            "patterns": _build_metadata_patterns(retrieval_query),
                        }
                    )
                continue

            logger.info(
                "retrieve_chunks lexical rows query=%r variant=%r rows=%s patterns=%s",
                query,
                retrieval_query,
                len(variant_rows),
                len(_build_metadata_patterns(retrieval_query)),
            )
            filtered_variant_rows, geography_mode, rejected_rows = _filter_rows_by_geography(
                variant_rows,
                profile,
            )
            if debug is not None:
                debug["branches"].append(
                    {
                        "stage": "chunk_candidates_lexical",
                        "variant": retrieval_query,
                        "rows": len(filtered_variant_rows),
                        "geography_mode": geography_mode,
                        "rejected_out_of_scope": rejected_rows[:10],
                        "patterns": _build_metadata_patterns(retrieval_query),
                        "sample_titles": [
                            row.get("title") or row.get("canonical_title")
                            for row in filtered_variant_rows[:5]
                        ],
                    }
                )
            adjusted_rows = []
            for row in filtered_variant_rows:
                adjusted_row = dict(row)
                adjusted_row["_distance"] = float(row.get("_distance", 0.35 + 0.03 * variant_index))
                adjusted_rows.append(adjusted_row)
            lexical_rows = _merge_candidate_rows(lexical_rows, adjusted_rows)

        if lexical_rows:
            chunks, documents = _select_documents_and_chunks(
                lexical_rows,
                profile,
                limit=limit,
                max_documents=6,
            )
            logger.info(
                "retrieve_chunks lexical selection query=%r rows=%s chunks=%s documents=%s",
                query,
                len(lexical_rows),
                len(chunks),
                len(documents),
            )
            if debug is not None:
                debug["selected"]["documents"] = [document.model_dump() for document in documents]
                debug["selected"]["chunks"] = [
                    {
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.chunk_id,
                        "title": chunk.title,
                        "section_title": chunk.section_title,
                        "year": chunk.year,
                        "url": chunk.url,
                    }
                    for chunk in chunks
                ]
                debug["selected"]["path"] = "chunk_lexical"
                debug["selected"]["geography_mode"] = _allowed_geography_classes(
                    lexical_rows, profile
                )[1]
                debug["selected"]["chunk_candidates"] = [
                    _chunk_score_breakdown(row, profile)
                    for row in lexical_rows[:10]
                ]
            if chunks or documents:
                return chunks, documents

        for variant_index, retrieval_query in enumerate(retrieval_queries):
            try:
                variant_rows = await asyncio.wait_for(
                    fetch_variant_rows(retrieval_query),
                    timeout=per_variant_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "retrieve_chunks variant timeout query=%r variant=%r timeout=%ss",
                    query,
                    retrieval_query,
                    per_variant_timeout,
                )
                if debug is not None:
                    debug["branches"].append(
                        {
                            "stage": "chunk_candidates_vector",
                            "variant": retrieval_query,
                            "status": "timeout",
                            "patterns": _build_metadata_patterns(retrieval_query),
                        }
                    )
                continue
            except Exception as error:
                logger.warning(
                    "retrieve_chunks variant failed query=%r variant=%r error=%s",
                    query,
                    retrieval_query,
                    error,
                )
                if debug is not None:
                    debug["branches"].append(
                        {
                            "stage": "chunk_candidates_vector",
                            "variant": retrieval_query,
                            "status": "error",
                            "error": str(error),
                            "patterns": _build_metadata_patterns(retrieval_query),
                        }
                    )
                continue

            logger.info(
                "retrieve_chunks variant rows query=%r variant=%r rows=%s",
                query,
                retrieval_query,
                len(variant_rows),
            )
            filtered_variant_rows, geography_mode, rejected_rows = _filter_rows_by_geography(
                variant_rows,
                profile,
            )
            if debug is not None:
                debug["branches"].append(
                    {
                        "stage": "chunk_candidates_vector",
                        "variant": retrieval_query,
                        "rows": len(filtered_variant_rows),
                        "geography_mode": geography_mode,
                        "rejected_out_of_scope": rejected_rows[:10],
                        "patterns": _build_metadata_patterns(retrieval_query),
                        "sample_titles": [
                            row.get("title") or row.get("canonical_title")
                            for row in filtered_variant_rows[:5]
                        ],
                    }
                )

            adjusted_rows: list[dict] = []
            for row in filtered_variant_rows:
                variant_distance = float(row.get("_distance", 1.0) or 1.0)
                adjusted_row = dict(row)
                adjusted_row["_distance"] = max(0.0, variant_distance + 0.03 * variant_index)
                adjusted_rows.append(adjusted_row)
            vector_rows = _merge_candidate_rows(vector_rows, adjusted_rows)

        if vector_rows:
            chunks, documents = _select_documents_and_chunks(
                vector_rows,
                profile,
                limit=limit,
                max_documents=6,
            )
            logger.info(
                "retrieve_chunks vector selection query=%r rows=%s chunks=%s documents=%s",
                query,
                len(vector_rows),
                len(chunks),
                len(documents),
            )
            if debug is not None:
                debug["selected"]["documents"] = [document.model_dump() for document in documents]
                debug["selected"]["chunks"] = [
                    {
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.chunk_id,
                        "title": chunk.title,
                        "section_title": chunk.section_title,
                        "year": chunk.year,
                        "url": chunk.url,
                    }
                    for chunk in chunks
                ]
                debug["selected"]["path"] = "chunk_vector"
                debug["selected"]["geography_mode"] = _allowed_geography_classes(
                    vector_rows, profile
                )[1]
                debug["selected"]["chunk_candidates"] = [
                    _chunk_score_breakdown(row, profile)
                    for row in vector_rows[:10]
                ]
            if chunks or documents:
                return chunks, documents

        logger.info(
            "retrieve_chunks fallback query=%r rows=%s",
            query,
            len(lexical_rows) + len(vector_rows),
        )
        chunks, documents = _select_documents_and_chunks(
            _merge_candidate_rows(lexical_rows, vector_rows),
            profile,
            limit=limit,
            max_documents=6,
        )
        if debug is not None:
            debug["selected"]["documents"] = [document.model_dump() for document in documents]
            debug["selected"]["chunks"] = [
                {
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "section_title": chunk.section_title,
                    "year": chunk.year,
                    "url": chunk.url,
                }
                for chunk in chunks
            ]
            debug["selected"]["path"] = "fallback"
            debug["selected"]["geography_mode"] = _allowed_geography_classes(
                _merge_candidate_rows(lexical_rows, vector_rows), profile
            )[1]
            debug["selected"]["chunk_candidates"] = [
                _chunk_score_breakdown(row, profile)
                for row in _merge_candidate_rows(lexical_rows, vector_rows)[:10]
            ]
        return chunks, documents

    async def get_sdg7_dataset(self) -> pd.DataFrame:
        """
        Get SDG 7 dataset.

        Returns
        -------
        pd.DataFrame
            Pandas data frame with SDG 7 indicators.
        """
        table = await self.connection.open_table("sdg7")
        df = await table.to_pandas()
        df.name = "indicators"
        return df

    async def get_knowledge_graph(self) -> nx.Graph:
        """
        Create a (weighted directed) knowledge graph from database tables.

        Returns
        -------
        nx.Graph
            Knowledge graph.
        """
        try:
            table = await self.connection.open_table("nodes")
        except ValueError as error:
            if "was not found" in str(error):
                return nx.DiGraph()
            raise
        df = await table.query().select(["name", "description", "weight"]).to_pandas()
        nodes = zip(
            df["name"].tolist(),
            df[["description", "weight"]].to_dict(orient="records"),
        )
        try:
            table = await self.connection.open_table("edges")
        except ValueError as error:
            if "was not found" in str(error):
                graph = nx.DiGraph()
                graph.add_nodes_from(nodes)
                return graph
            raise
        # check if level column exists in the table schema
        schema = await table.schema()
        edge_columns = ["subject", "object", "predicate", "description", "weight"]
        edge_attrs = ["predicate", "description", "weight"]
        if "level" in [field.name for field in schema]:
            edge_columns.append("level")
            edge_attrs.append("level")
        df = await table.query().select(edge_columns).to_pandas()
        edges = zip(
            df["subject"].tolist(),
            df["object"].tolist(),
            df[edge_attrs].to_dict(orient="records"),
        )
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph


# since the model uses the docstring, don't mention the artifacts there
@tool(parse_docstring=True, response_format="content_and_artifact")
async def retrieve_chunks(
    query: str, years: int | list[int] | None = None
) -> tuple[str, list[Document]]:
    """Retrieve relevant document chunks from the Sustainable Energy Academy database.

    The database can be used to answer questions on energy, climate change and
    sustainable development in general. Use the database to provide accurate and
    grounded responses. References are returned separately as structured document metadata,
    so do not include a source list or raw URLs in the answer body.

    Args:
        query (str): Plain text user query.
        years (Union[int, Tuple[int, ...], None]): Specific year or years the user refers
            to if applicable. If not provided, retrieval should infer whether recency
            should be preferred based on the query intent.

    Returns:
        str: JSON object containing the most relevant document chunks and references.
    """
    def normalize_years(value: int | list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if isinstance(value, int):
            return [value]
        clean = []
        for item in value:
            if isinstance(item, int):
                clean.append(item)
            elif isinstance(item, str) and item.isdigit():
                clean.append(int(item))
        return clean or None

    safe_years = normalize_years(years)
    timeout_seconds = _env_timeout_seconds(
        "RETRIEVE_CHUNKS_TIMEOUT_SECONDS",
        60.0,
    )
    connection = None
    try:
        connection = await get_connection()
        client = Client(connection)
        chunks, documents = await asyncio.wait_for(
            client.retrieve_chunks(query, safe_years),
            timeout=timeout_seconds,
        )
        data = json.dumps(
            {
                "chunks": [chunk.to_context() for chunk in chunks],
                "documents": [document.model_dump() for document in documents],
            }
        )
        return data, documents
    except asyncio.TimeoutError:
        logger.warning(
            "retrieve_chunks timed out after %ss for query=%r",
            timeout_seconds,
            query,
        )
        return "[]", []
    except Exception as error:
        logger.exception("retrieve_chunks tool failed: %s", error)
        # Return empty evidence payload so the agent can continue answering.
        return "[]", []
    finally:
        close = getattr(connection, "close", None)
        if callable(close):
            result = close()
            if isawaitable(result):
                with contextlib.suppress(Exception):
                    await result
