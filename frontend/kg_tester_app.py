"""
Standalone app for the KG tester UI.

This keeps the tester runtime separate from the production API app.
"""

import json
import os
import ipaddress
from typing import Literal
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="KG Tester Frontend", docs_url=None, redoc_url=None, openapi_url=None)
templates = Jinja2Templates(directory="frontend/templates")


class ProxyMessage(BaseModel):
    role: Literal["assistant", "human"]
    content: str = Field(min_length=0, max_length=16_384)


class ModelProxyRequest(BaseModel):
    target: Literal["local", "remote"] = "local"
    graph_version: Literal["default", "v1", "v2"] = "default"
    remote_base: str | None = None
    messages: list[ProxyMessage]


def _normalise_base(url: str) -> str:
    return (url or "").strip().rstrip("/")


def _validated_base(url: str) -> str:
    base = _normalise_base(url)
    parsed = urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail=f"Invalid API base URL: {url}")
    return base


def _get_local_base() -> str:
    return _validated_base(os.getenv("KG_TESTER_LOCAL_API_BASE_URL", "http://127.0.0.1:8000"))


def _get_remote_default_base() -> str:
    configured = os.getenv("KG_TESTER_REMOTE_API_BASE_URL", "")
    return _validated_base(configured) if configured.strip() else ""


def _get_api_base(target: str, remote_base: str | None) -> str:
    if target == "remote":
        candidate = remote_base or _get_remote_default_base()
        base = _validated_base(candidate) if candidate else ""
        if not base:
            raise HTTPException(
                status_code=400,
                detail="Remote API base URL is required for remote target.",
            )
        return base
    return _get_local_base()


def _get_api_key() -> str:
    api_key = (os.getenv("KG_TESTER_API_KEY") or os.getenv("API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Missing KG_TESTER_API_KEY (or API_KEY) in environment.",
        )
    return api_key


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
    """
    Restrict tester usage to loopback clients only.

    The tester app proxies backend API calls with a server-side key, so it should
    not be exposed beyond local development.
    """
    client_host = request.client.host if request.client else None
    if not _is_loopback_client(client_host):
        raise HTTPException(status_code=403, detail="KG tester is restricted to local requests.")


@app.get("/", include_in_schema=False)
async def root(request: Request):
    _require_local_request(request)
    return RedirectResponse(url="/kg-tester", status_code=307)


@app.get("/kg-tester", include_in_schema=False)
async def kg_tester_page(request: Request):
    _require_local_request(request)
    return templates.TemplateResponse(
        request=request,
        name="kg_tester.html",
        context={"remote_api_base_url": _get_remote_default_base()},
    )


@app.post("/kg-tester/api/model", include_in_schema=False)
async def proxy_model(request: Request, payload: ModelProxyRequest):
    _require_local_request(request)
    api_base = _get_api_base(payload.target, payload.remote_base)
    api_key = _get_api_key()
    path = "/model" if payload.graph_version == "default" else f"/model?graph_version={payload.graph_version}"
    upstream_url = f"{api_base}{path}"
    request_headers = {
        "Content-Type": "application/json",
        "X-Api-Key": api_key,
    }

    client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=None, write=60))
    try:
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
        raise HTTPException(status_code=502, detail=f"Failed to reach upstream API: {error}") from error

    if upstream.status_code >= 400:
        try:
            payload = json.loads((await upstream.aread()).decode("utf-8"))
            detail = payload.get("detail", f"Upstream error ({upstream.status_code})")
        except Exception:
            detail = f"Upstream error ({upstream.status_code})"
        await upstream.aclose()
        await client.aclose()
        raise HTTPException(status_code=upstream.status_code, detail=detail)

    response_headers = {}
    for header in ("X-Request-Id", "X-KG-Timing", "Server-Timing"):
        value = upstream.headers.get(header)
        if value:
            response_headers[header] = value

    async def stream_upstream():
        try:
            async for chunk in upstream.aiter_bytes():
                if chunk:
                    yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    media_type = upstream.headers.get("content-type", "application/x-ndjson")
    return StreamingResponse(
        content=stream_upstream(),
        status_code=upstream.status_code,
        media_type=media_type,
        headers=response_headers,
    )
