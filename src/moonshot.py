"""FastAPI Moonshot routes and Azure OpenAI helpers."""

from __future__ import annotations

import ipaddress
import json
import os
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from time import monotonic
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from openai import APIError, APIStatusError, AzureOpenAI, OpenAI

from .moonshot_models import (
    MoonshotHealthResponse,
    ParseQueryRequest,
    ParseQueryResponse,
    ProjectSynopsisRequest,
    ProjectSynopsisResponse,
)

router = APIRouter(prefix="/api/moonshot", tags=["moonshot"])

FILTER_KEYS = (
    "funding",
    "genderMarker",
    "category",
    "subCategory",
    "bureau",
    "economy",
    "hdiTier",
    "specialGrouping",
    "continentRegion",
    "subRegion",
    "sahel",
    "crisis",
    "countryCode",
)
MAX_DESCRIPTION_LENGTH = 500
PLACEHOLDER_VALUES = {
    "your_azure_openai_key_here",
    "your-resource-name.openai.azure.com",
    "https://your-resource-name.openai.azure.com/",
    "your_openai_api_key_here",
}
RATE_LIMIT_LOCK = Lock()
RATE_LIMIT_BUCKETS: dict[tuple[str, str], deque[float]] = {}


@dataclass(frozen=True)
class MoonshotSettings:
    azure_openai_key: str | None
    azure_openai_endpoint: str | None
    azure_openai_api_version: str
    parse_model: str
    synopsis_model: str
    openai_api_key: str | None
    openai_parse_model: str
    openai_synopsis_model: str

    @property
    def provider(self) -> str | None:
        if self.has_azure_config:
            return "azure"
        if self.has_openai_config:
            return "openai"
        return None

    @property
    def configured(self) -> bool:
        return self.provider is not None

    @property
    def has_azure_config(self) -> bool:
        return bool(
            self.azure_openai_key
            and self.azure_openai_endpoint
            and not is_placeholder_config_value(self.azure_openai_key)
            and not is_placeholder_config_value(self.azure_openai_endpoint)
        )

    @property
    def has_openai_config(self) -> bool:
        return bool(
            self.openai_api_key
            and not is_placeholder_config_value(self.openai_api_key)
        )

    @property
    def active_parse_model(self) -> str | None:
        if self.provider == "azure":
            return self.parse_model
        if self.provider == "openai":
            return self.openai_parse_model
        return None

    @property
    def active_synopsis_model(self) -> str | None:
        if self.provider == "azure":
            return self.synopsis_model
        if self.provider == "openai":
            return self.openai_synopsis_model
        return None


@lru_cache
def get_settings() -> MoonshotSettings:
    parse_model = (
        os.getenv("AZURE_OPENAI_MOONSHOT_PARSE_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_PARSE_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_CHAT_MODEL")
        or "gpt-4.1-mini"
    ).strip()
    synopsis_model = (
        os.getenv("AZURE_OPENAI_MOONSHOT_SYNOPSIS_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_SYNOPSIS_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_CHAT_MODEL")
        or "gpt-4.1-mini"
    ).strip()
    openai_parse_model = (os.getenv("OPENAI_PARSE_MODEL") or parse_model).strip()
    openai_synopsis_model = (os.getenv("OPENAI_SYNOPSIS_MODEL") or synopsis_model).strip()

    return MoonshotSettings(
        azure_openai_key=normalize_string(os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY")) or None,
        azure_openai_endpoint=normalize_string(os.getenv("AZURE_OPENAI_ENDPOINT")) or None,
        azure_openai_api_version=normalize_string(os.getenv("AZURE_OPENAI_API_VERSION")) or "2025-03-01-preview",
        parse_model=parse_model,
        synopsis_model=synopsis_model,
        openai_api_key=normalize_string(os.getenv("OPENAI_API_KEY")) or None,
        openai_parse_model=openai_parse_model,
        openai_synopsis_model=openai_synopsis_model,
    )


@lru_cache
def get_openai_client() -> AzureOpenAI | OpenAI | None:
    settings = get_settings()
    if settings.provider == "azure":
        return AzureOpenAI(
            api_key=settings.azure_openai_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
    if settings.provider == "openai":
        return OpenAI(api_key=settings.openai_api_key)
    return None


def get_allowed_origins() -> list[str]:
    raw_origins = normalize_string(os.getenv("ALLOWED_ORIGINS"))
    if not raw_origins:
        return []
    normalized_origins = []
    for origin in raw_origins.split(","):
        normalized = normalize_origin(origin)
        if normalized:
            normalized_origins.append(normalized)
    return normalized_origins


def normalize_origin(value: str | None) -> str:
    normalized = normalize_string(value).strip("\"'")
    return normalized.rstrip("/")


def build_cors_headers(request: Request) -> dict[str, str]:
    origin = normalize_origin(request.headers.get("origin"))
    allowed_origins = get_allowed_origins()

    if allowed_origins:
        if origin and origin in allowed_origins:
            return {
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type,Authorization",
                "Vary": "Origin",
            }
        return {}

    if origin:
        return {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
        }

    return {}


def build_cors_headers(request: Request) -> dict[str, str]:
    origin = normalize_string(request.headers.get("origin"))
    allowed_origins = get_allowed_origins()

    if allowed_origins:
        if origin and origin in allowed_origins:
            return {
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type,Authorization",
                "Vary": "Origin",
            }
        return {}

    if origin:
        return {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
        }

    return {}


@dataclass(frozen=True)
class MoonshotRateLimitSettings:
    window_seconds: int
    parse_limit: int
    synopsis_limit: int


@lru_cache
def get_rate_limit_settings() -> MoonshotRateLimitSettings:
    return MoonshotRateLimitSettings(
        window_seconds=max(1, int(normalize_string(os.getenv("MOONSHOT_RATE_LIMIT_WINDOW_SECONDS")) or "300")),
        parse_limit=max(1, int(normalize_string(os.getenv("MOONSHOT_PARSE_RATE_LIMIT")) or "20")),
        synopsis_limit=max(1, int(normalize_string(os.getenv("MOONSHOT_SYNOPSIS_RATE_LIMIT")) or "8")),
    )


def is_placeholder_config_value(value: str | None) -> bool:
    normalized = normalize_string(value).lower()
    if not normalized:
        return True
    if normalized in {item.lower() for item in PLACEHOLDER_VALUES}:
        return True
    return normalized.startswith("your_") or "your-resource-name" in normalized


def require_string(value: str | None, field_name: str) -> str:
    result = normalize_string(value)
    if not result:
        raise HTTPException(status_code=400, detail=f"{field_name} is required.")
    return result


def require_openai_client() -> AzureOpenAI | OpenAI:
    client = get_openai_client()
    if client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Moonshot API credentials are not configured. "
                "Set Azure OpenAI environment variables on the API server."
            ),
        )
    return client


def get_client_identifier(request: Request) -> str:
    forwarded_for = normalize_string(request.headers.get("x-forwarded-for"))
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def is_loopback_client(identifier: str) -> bool:
    candidate = identifier.strip().lower()
    if candidate in {"localhost", "testclient"}:
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def is_allowed_request_origin(request: Request) -> bool:
    allowed_origins = get_allowed_origins()
    if not allowed_origins:
        return True

    origin = normalize_origin(request.headers.get("origin"))
    if origin and origin in allowed_origins:
        return True

    referer = normalize_string(request.headers.get("referer")).strip("\"'")
    if referer:
        for allowed_origin in allowed_origins:
            prefix = allowed_origin
            if referer == prefix or referer.startswith(f"{prefix}/"):
                return True

    return is_loopback_client(get_client_identifier(request))


def enforce_allowed_request_origin(request: Request) -> None:
    if not is_allowed_request_origin(request):
        raise HTTPException(
            status_code=403,
            detail="Moonshot API is restricted to configured browser origins.",
        )


def enforce_rate_limit(request: Request, endpoint_name: str) -> None:
    settings = get_rate_limit_settings()
    limit = settings.parse_limit if endpoint_name == "parse" else settings.synopsis_limit
    client_identifier = get_client_identifier(request)
    bucket_key = (endpoint_name, client_identifier)
    now = monotonic()
    cutoff = now - settings.window_seconds

    with RATE_LIMIT_LOCK:
        bucket = RATE_LIMIT_BUCKETS.setdefault(bucket_key, deque())
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= limit:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Moonshot {endpoint_name} rate limit exceeded. "
                    f"Please wait before sending more requests."
                ),
            )

        bucket.append(now)


def clear_runtime_state() -> None:
    get_settings.cache_clear()
    get_openai_client.cache_clear()
    get_rate_limit_settings.cache_clear()
    with RATE_LIMIT_LOCK:
        RATE_LIMIT_BUCKETS.clear()


def raise_provider_http_error(error: Exception) -> None:
    if isinstance(error, HTTPException):
        raise error

    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)

    message = normalize_string(getattr(error, "message", None)) or normalize_string(str(error))
    lower_message = message.lower()

    if status_code == 401:
        raise HTTPException(
            status_code=503,
            detail=(
                "Moonshot API credentials were rejected by Azure OpenAI. "
                "Update api/.env with a real key and deployment values."
            ),
        ) from error

    if status_code == 404 and ("deployment" in lower_message or "resource not found" in lower_message):
        raise HTTPException(
            status_code=503,
            detail=(
                "The configured Azure OpenAI deployment was not found. "
                "Check the Moonshot deployment names in api/.env."
            ),
        ) from error

    if status_code == 429:
        raise HTTPException(
            status_code=503,
            detail="Azure OpenAI rate-limited the Moonshot request. Please try again.",
        ) from error

    if isinstance(error, (APIStatusError, APIError)) and message:
        raise HTTPException(status_code=503, detail=message) from error

    raise HTTPException(status_code=500, detail=message or "Unexpected Moonshot API error.") from error


def normalize_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def clamp_string(value: Any, length: int = MAX_DESCRIPTION_LENGTH) -> str:
    return normalize_string(value)[:length]


def is_plain_object(value: Any) -> bool:
    return isinstance(value, dict)


def sanitize_filter_catalog(filter_catalog: Any) -> dict[str, list[dict[str, Any]]]:
    if not is_plain_object(filter_catalog) or not is_plain_object(filter_catalog.get("optionsByKey")):
        return {key: [] for key in FILTER_KEYS}

    options_by_key = filter_catalog["optionsByKey"]
    sanitized: dict[str, list[dict[str, Any]]] = {}
    for key in FILTER_KEYS:
        raw_options = options_by_key.get(key, [])
        options: list[dict[str, Any]] = []
        if isinstance(raw_options, list):
            for option in raw_options[:500]:
                if not is_plain_object(option):
                    continue
                value = normalize_string(option.get("value"))
                label = normalize_string(option.get("label")) or value
                aliases = option.get("aliases", [])
                normalized_aliases = []
                if isinstance(aliases, list):
                    normalized_aliases = [normalize_string(alias) for alias in aliases if normalize_string(alias)][:8]
                if value:
                    options.append(
                        {
                            "value": value,
                            "label": label,
                            "aliases": normalized_aliases,
                        }
                    )
        sanitized[key] = options
    return sanitized


def sanitize_parsed_filters(filters: Any, filter_catalog: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    if not is_plain_object(filters):
        return {}

    sanitized: dict[str, str] = {}
    for key in FILTER_KEYS:
        value = normalize_string(filters.get(key))
        allowed_values = {"all", *(option["value"] for option in filter_catalog.get(key, []))}
        if value and value != "all" and value in allowed_values:
            sanitized[key] = value
    return sanitized


def sanitize_unresolved_terms(terms: Any) -> list[str]:
    if not isinstance(terms, list):
        return []
    result = []
    for term in terms:
        normalized = normalize_string(term)
        if normalized:
            result.append(normalized)
    return result[:12]


def build_parse_query_schema(filter_catalog: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "name": "moonshot_filters",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "filters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        key: {
                            "type": "string",
                            "enum": ["all", *(option["value"] for option in filter_catalog.get(key, []))],
                        }
                        for key in FILTER_KEYS
                    },
                    "required": list(FILTER_KEYS),
                },
                "unresolvedTerms": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["filters", "unresolvedTerms"],
        },
    }


def build_catalog_summary(filter_catalog: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {
        key: [
            {
                "value": option["value"],
                "label": option["label"],
                "aliases": option["aliases"],
            }
            for option in filter_catalog.get(key, [])
        ]
        for key in FILTER_KEYS
    }


def sanitize_summary_metrics(summary_metrics: Any) -> dict[str, Any] | None:
    if not is_plain_object(summary_metrics):
        return None

    numeric_fields = (
        "projectCount",
        "countryCount",
        "totalBudget",
        "directBeneficiaries",
        "vfBeneficiaries",
        "nonVfBeneficiaries",
        "cleanElectricityBeneficiaries",
        "cleanCookingBeneficiaries",
        "productiveUseBeneficiaries",
        "policyProjectCount",
    )

    sanitized = {field: coerce_number(summary_metrics.get(field)) for field in numeric_fields}
    top_categories = summary_metrics.get("topBeneficiaryCategories", [])
    sanitized["topBeneficiaryCategories"] = []
    if isinstance(top_categories, list):
        for entry in top_categories[:5]:
            if not is_plain_object(entry):
                continue
            category = clamp_string(entry.get("category"), 120)
            if not category:
                continue
            sanitized["topBeneficiaryCategories"].append(
                {
                    "category": category,
                    "value": coerce_number(entry.get("value")),
                }
            )
    return sanitized


def sanitize_project_context(project_context: Any) -> dict[str, Any] | None:
    if not is_plain_object(project_context):
        return None

    top_projects = project_context.get("topProjects", [])
    sanitized_projects = []
    if isinstance(top_projects, list):
        for project in top_projects[:10]:
            if not is_plain_object(project):
                continue
            categories = project.get("primaryOutputCategories", [])
            normalized_categories = []
            if isinstance(categories, list):
                normalized_categories = [clamp_string(category, 120) for category in categories if clamp_string(category, 120)][:5]
            sanitized_projects.append(
                {
                    "id": clamp_string(project.get("id"), 80),
                    "title": clamp_string(project.get("title"), 200),
                    "countryName": clamp_string(project.get("countryName"), 120),
                    "description": clamp_string(project.get("description"), MAX_DESCRIPTION_LENGTH),
                    "budget": coerce_number(project.get("budget")),
                    "directBeneficiaries": coerce_number(project.get("directBeneficiaries")),
                    "primaryOutputCategories": normalized_categories,
                }
            )
    return {
        "totalProjects": int(coerce_number(project_context.get("totalProjects"))),
        "topProjects": sanitized_projects,
    }


def sanitize_filters_payload(filters: Any) -> dict[str, str]:
    if not is_plain_object(filters):
        return {}

    sanitized = {}
    for key in FILTER_KEYS:
        value = normalize_string(filters.get(key))
        if value:
            sanitized[key] = value
    return sanitized


def coerce_number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def get_chat_text(completion: Any) -> str:
    content = completion.choices[0].message.content
    if isinstance(content, str):
        return normalize_string(content)
    if isinstance(content, list):
        text_parts = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                text_parts.append(text)
        return normalize_string(" ".join(text_parts))
    return ""


def create_structured_chat_completion(
    *,
    client: AzureOpenAI | OpenAI,
    model: str,
    schema: dict[str, Any],
    system_prompt: str,
    user_payload: dict[str, Any],
) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": schema,
        },
    )
    return get_chat_text(completion)


def create_text_chat_completion(
    *,
    client: AzureOpenAI | OpenAI,
    model: str,
    system_prompt: str,
    user_payload: dict[str, Any],
) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
    )
    return get_chat_text(completion)


def generate_parsed_filters(
    *,
    client: AzureOpenAI | OpenAI,
    model: str,
    schema: dict[str, Any],
    system_prompt: str,
    user_payload: dict[str, Any],
) -> dict[str, Any]:
    output_text = create_structured_chat_completion(
        client=client,
        model=model,
        schema=schema,
        system_prompt=system_prompt,
        user_payload=user_payload,
    )
    if not output_text:
        raise HTTPException(status_code=502, detail="Model returned an empty parse-query response.")
    try:
        return json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="Model returned invalid parse-query JSON.") from exc


def generate_project_synopsis(
    *,
    client: AzureOpenAI | OpenAI,
    model: str,
    system_prompt: str,
    user_payload: dict[str, Any],
) -> str:
    synopsis = create_text_chat_completion(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_payload=user_payload,
    )
    return synopsis or "No synopsis was generated."


@router.get("/health", response_model=MoonshotHealthResponse)
def health(request: Request) -> JSONResponse:
    settings = get_settings()
    payload = MoonshotHealthResponse(
        configured=settings.configured,
        provider=settings.provider,
        parseModel=settings.active_parse_model,
        synopsisModel=settings.active_synopsis_model,
    )
    return JSONResponse(content=payload.model_dump(mode="json"), headers=build_cors_headers(request))


@router.options("/health", include_in_schema=False)
def health_options(request: Request) -> Response:
    return Response(status_code=204, headers=build_cors_headers(request))


@router.post("/parse-query", response_model=ParseQueryResponse)
def parse_query(payload: ParseQueryRequest, request: Request) -> JSONResponse:
    try:
        enforce_allowed_request_origin(request)
        enforce_rate_limit(request, "parse")
        client = require_openai_client()
        settings = get_settings()
        query = require_string(payload.query, "query")
        locale = normalize_string(payload.locale) or "en"
        filter_catalog = sanitize_filter_catalog(payload.filterCatalog)
        schema = build_parse_query_schema(filter_catalog)
        system_prompt = " ".join(
            [
                "You convert natural-language dashboard questions into structured filters.",
                "Only map terms that are explicitly present in the user query.",
                "Do not infer semantic topics, latent intent, project themes, or free-text retrieval terms.",
                "Use only filter values that exist in the provided filter catalog.",
                'Return "all" for any filter dimension that is not explicitly matched by the query.',
                "Return unsupported or leftover topical terms in unresolvedTerms.",
                "If the query mentions a specific country, return the matching countryCode filter.",
                "If nothing maps to a known filter, return an empty filters object after sanitization.",
                f"User locale: {locale}.",
            ]
        )
        user_payload = {
            "query": query,
            "filterCatalog": build_catalog_summary(filter_catalog),
        }
        parsed = generate_parsed_filters(
            client=client,
            model=settings.active_parse_model or settings.parse_model,
            schema=schema,
            system_prompt=system_prompt,
            user_payload=user_payload,
        )
        payload = ParseQueryResponse(
            filters=sanitize_parsed_filters(parsed.get("filters"), filter_catalog),
            unresolvedTerms=sanitize_unresolved_terms(parsed.get("unresolvedTerms")),
        )
        return JSONResponse(content=payload.model_dump(mode="json"), headers=build_cors_headers(request))
    except Exception as error:
        raise_provider_http_error(error)


@router.options("/parse-query", include_in_schema=False)
def parse_query_options(request: Request) -> Response:
    enforce_allowed_request_origin(request)
    return Response(status_code=204, headers=build_cors_headers(request))


@router.post("/project-synopsis", response_model=ProjectSynopsisResponse)
def project_synopsis(payload: ProjectSynopsisRequest, request: Request) -> JSONResponse:
    try:
        enforce_allowed_request_origin(request)
        enforce_rate_limit(request, "synopsis")
        client = require_openai_client()
        settings = get_settings()
        query = require_string(payload.query, "query")
        locale = normalize_string(payload.locale) or "en"
        filters = sanitize_filters_payload(payload.filters)
        summary_metrics = sanitize_summary_metrics(payload.summaryMetrics)
        project_context = sanitize_project_context(payload.projectContext)

        if not summary_metrics or not project_context:
            raise HTTPException(status_code=400, detail="summaryMetrics and projectContext are required.")

        if not project_context["totalProjects"]:
            empty_payload = ProjectSynopsisResponse(
                synopsis="No matching projects are available for an AI-generated project overview."
            )
            return JSONResponse(
                content=empty_payload.model_dump(mode="json"),
                headers=build_cors_headers(request),
            )

        system_prompt = " ".join(
            [
                "Write a concise overview of the filtered UNDP energy project set.",
                "Use only the structured metrics and project context provided.",
                "Do not invent countries, projects, budgets, beneficiaries, outputs, or causal claims.",
                "Do not restate the deterministic summary verbatim.",
                "Focus on project mix, recurring themes, and representative examples from the supplied top projects.",
                "Do not output bullet points, markdown headings, or a project title list.",
                "Keep the answer to one short paragraph, about 80 to 140 words.",
                f"User locale: {locale}.",
            ]
        )
        synopsis = generate_project_synopsis(
            client=client,
            model=settings.active_synopsis_model or settings.synopsis_model,
            system_prompt=system_prompt,
            user_payload={
                "query": query,
                "filters": filters,
                "summaryMetrics": summary_metrics,
                "projectContext": project_context,
            },
        )
        response_payload = ProjectSynopsisResponse(synopsis=synopsis)
        return JSONResponse(
            content=response_payload.model_dump(mode="json"),
            headers=build_cors_headers(request),
        )
    except Exception as error:
        raise_provider_http_error(error)


@router.options("/project-synopsis", include_in_schema=False)
def project_synopsis_options(request: Request) -> Response:
    enforce_allowed_request_origin(request)
    return Response(status_code=204, headers=build_cors_headers(request))
