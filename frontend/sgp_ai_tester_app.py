"""
Standalone local tester UI for the SGP AI RAG assistant.

This keeps the SGP tester separate from the production API app. It proxies
requests to the backend with a server-side API key so browser JavaScript never
receives credentials.
"""

from __future__ import annotations

import ipaddress
import json
import os
from typing import Annotated, Literal
from urllib.parse import quote, urlparse
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="SGP AI Tester Frontend", docs_url=None, redoc_url=None, openapi_url=None)
templates = Jinja2Templates(directory="frontend/templates")

DEFAULT_ASSISTANT_ID = "sgp_ai"
DEFAULT_LOCAL_API_BASE_URL = "http://127.0.0.1:8016"
DEFAULT_BACKEND_API_BASE_URL = "https://sea-ai-api.azurewebsites.net"
TargetMode = Literal["local", "backend"]


class ProxyMessage(BaseModel):
    role: Literal["assistant", "human"]
    content: str = Field(min_length=0, max_length=16_384)


class ModelProxyRequest(BaseModel):
    messages: list[ProxyMessage] = Field(min_length=1)
    target_mode: TargetMode = "local"


def _normalise_base(url: str) -> str:
    return (url or "").strip().rstrip("/")


def _validated_base(url: str) -> str:
    base = _normalise_base(url)
    parsed = urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail=f"Invalid API base URL: {url}")
    return base


def _get_api_base(mode: TargetMode) -> str:
    if mode == "backend":
        return _validated_base(os.getenv("SGP_AI_BACKEND_API_BASE_URL") or DEFAULT_BACKEND_API_BASE_URL)
    return _validated_base(os.getenv("SGP_AI_LOCAL_API_BASE_URL") or DEFAULT_LOCAL_API_BASE_URL)


def _get_assistant_id() -> str:
    assistant_id = (os.getenv("SGP_TESTER_ASSISTANT_ID") or DEFAULT_ASSISTANT_ID).strip()
    if not assistant_id:
        raise HTTPException(status_code=500, detail="Missing SGP_TESTER_ASSISTANT_ID.")
    return assistant_id


def _get_api_key(mode: TargetMode) -> str:
    if mode == "backend":
        api_key = (os.getenv("SGP_AI_BACKEND_API_KEY") or os.getenv("API_KEY") or "").strip()
        env_hint = "SGP_AI_BACKEND_API_KEY or API_KEY"
    else:
        api_key = (os.getenv("SGP_AI_LOCAL_API_KEY") or os.getenv("API_KEY") or "").strip()
        env_hint = "SGP_AI_LOCAL_API_KEY or API_KEY"
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail=f"Missing {env_hint} in environment.",
        )
    return api_key


def _headers(request: Request, mode: TargetMode) -> dict[str, str]:
    request_id = request.headers.get("X-Request-Id") or f"sgp-ai-tester-{uuid4().hex}"
    return {
        "Accept": "application/json, application/x-ndjson",
        "Content-Type": "application/json",
        "X-Api-Key": _get_api_key(mode),
        "X-Request-Id": request_id,
    }


def _is_loopback_client(host: str | None) -> bool:
    if not host:
        return False
    candidate = host.strip().lower()
    if candidate in {"localhost", "testclient"}:
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def _require_local_request(request: Request) -> None:
    client_host = request.client.host if request.client else None
    if not _is_loopback_client(client_host):
        raise HTTPException(status_code=403, detail="SGP AI tester is restricted to local requests.")


def _json_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        text = response.text.strip()
        return text or f"Upstream error ({response.status_code})"
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("error")
        if isinstance(detail, str):
            return detail
        if detail is not None:
            return json.dumps(detail)
    return f"Upstream error ({response.status_code})"


async def _raise_for_upstream(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    raise HTTPException(status_code=response.status_code, detail=_json_error_detail(response))


def _copy_trace_headers(response: httpx.Response) -> dict[str, str]:
    copied = {}
    for header in ("X-Request-Id", "X-KG-Timing", "Server-Timing"):
        value = response.headers.get(header)
        if value:
            copied[header] = value
    return copied


async def _ensure_corpus_ready(
    client: httpx.AsyncClient,
    request: Request,
    api_base: str,
    assistant_id: str,
    mode: TargetMode,
) -> dict[str, str]:
    documents_url = f"{api_base}/assistants/{quote(assistant_id, safe='')}/documents"
    response = await client.get(
        documents_url,
        headers=_headers(request, mode),
        params={"limit": 1},
    )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=409,
            detail=f"SGP AI corpus is not ready: {_json_error_detail(response)}",
        )
    payload = response.json()
    if not isinstance(payload, list):
        raise HTTPException(status_code=502, detail="Backend documents response was not a list.")
    if not payload:
        raise HTTPException(
            status_code=409,
            detail="SGP AI corpus is not imported yet. Refresh storage credentials and import the corpus before querying.",
        )
    return _copy_trace_headers(response)


@app.get("/", include_in_schema=False)
async def root(request: Request):
    _require_local_request(request)
    return RedirectResponse(url="/sgp-ai-tester", status_code=307)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon(request: Request):
    _require_local_request(request)
    return FileResponse("static/favicon.ico")


@app.get("/sgp-ai-tester", include_in_schema=False)
async def sgp_ai_tester_page(request: Request):
    _require_local_request(request)
    return templates.TemplateResponse(
        request=request,
        name="sgp_ai_tester.html",
        context={
            "assistant_id": _get_assistant_id(),
            "targets": {
                "local": _get_api_base("local"),
                "backend": _get_api_base("backend"),
            },
        },
    )


@app.get("/sgp-ai-tester/api/status", include_in_schema=False)
async def proxy_status(request: Request, mode: TargetMode = "local"):
    _require_local_request(request)
    api_base = _get_api_base(mode)
    assistant_id = _get_assistant_id()
    trace_headers: dict[str, str] = {}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0)) as client:
            response = await client.get(f"{api_base}/assistants", headers=_headers(request, mode))
            trace_headers.update(_copy_trace_headers(response))
            await _raise_for_upstream(response)
            assistants = response.json()
            if not isinstance(assistants, list):
                raise HTTPException(status_code=502, detail="Backend /assistants response was not a list.")
            profile = next(
                (item for item in assistants if isinstance(item, dict) and item.get("assistant_id") == assistant_id),
                None,
            )
    except HTTPException:
        raise
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"Failed to reach backend API: {error}") from error

    return JSONResponse(
        {
            "ok": True,
            "mode": mode,
            "backend_base_url": api_base,
            "assistant_id": assistant_id,
            "installed": profile is not None,
            "profile": profile,
            "assistant_count": len(assistants),
        },
        headers=trace_headers,
    )


@app.get("/sgp-ai-tester/api/retrieve", include_in_schema=False)
async def proxy_retrieve(
    request: Request,
    query: Annotated[str, Query(min_length=2)],
    limit: Annotated[int, Query(ge=1, le=50)] = 12,
    mode: TargetMode = "local",
):
    _require_local_request(request)
    api_base = _get_api_base(mode)
    assistant_id = quote(_get_assistant_id(), safe="")
    url = f"{api_base}/assistants/{assistant_id}/debug/retrieve"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=10.0)) as client:
            trace_headers = await _ensure_corpus_ready(client, request, api_base, _get_assistant_id(), mode)
            response = await client.get(
                url,
                headers=_headers(request, mode),
                params={"query": query, "limit": limit},
            )
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"Failed to reach backend API: {error}") from error
    await _raise_for_upstream(response)
    trace_headers.update(_copy_trace_headers(response))
    return JSONResponse(response.json(), headers=trace_headers)


@app.post("/sgp-ai-tester/api/model", include_in_schema=False)
async def proxy_model(request: Request, payload: ModelProxyRequest):
    _require_local_request(request)
    if payload.messages[-1].role != "human":
        raise HTTPException(status_code=400, detail="The last message must come from the user.")

    mode = payload.target_mode
    api_base = _get_api_base(mode)
    assistant_id = quote(_get_assistant_id(), safe="")
    upstream_url = f"{api_base}/assistants/{assistant_id}/model"
    request_headers = _headers(request, mode)
    request_headers["Accept"] = "application/x-ndjson"

    client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0, read=None))
    try:
        trace_headers = await _ensure_corpus_ready(client, request, api_base, _get_assistant_id(), mode)
        upstream = await client.send(
            client.build_request(
                method="POST",
                url=upstream_url,
                headers=request_headers,
                json=[message.model_dump() for message in payload.messages],
            ),
            stream=True,
        )
    except httpx.HTTPError as error:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"Failed to reach backend API: {error}") from error

    if upstream.status_code >= 400:
        body = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        try:
            detail_payload = json.loads(body.decode("utf-8"))
            detail = detail_payload.get("detail") if isinstance(detail_payload, dict) else None
        except Exception:
            detail = body.decode("utf-8", errors="replace").strip()
        raise HTTPException(
            status_code=upstream.status_code,
            detail=detail or f"Upstream error ({upstream.status_code})",
        )

    response_headers = trace_headers
    response_headers.update(_copy_trace_headers(upstream))
    if "X-Request-Id" not in response_headers:
        response_headers["X-Request-Id"] = request_headers["X-Request-Id"]

    async def stream_upstream():
        try:
            async for chunk in upstream.aiter_bytes():
                if chunk:
                    yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        content=stream_upstream(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/x-ndjson"),
        headers=response_headers,
    )
