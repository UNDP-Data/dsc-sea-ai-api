"""Tests for the standalone SGP AI tester proxy."""

from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

import frontend.sgp_ai_tester_app as app_module


REAL_ASYNC_CLIENT = httpx.AsyncClient


def _patch_async_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return REAL_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(app_module.httpx, "AsyncClient", factory)


def _set_env(monkeypatch):
    monkeypatch.setenv("SGP_AI_LOCAL_API_KEY", "local-key")
    monkeypatch.setenv("SGP_AI_BACKEND_API_KEY", "backend-key")
    monkeypatch.setenv("SGP_AI_LOCAL_API_BASE_URL", "http://local.test")
    monkeypatch.setenv("SGP_AI_BACKEND_API_BASE_URL", "http://backend.test")
    monkeypatch.setenv("SGP_TESTER_ASSISTANT_ID", "sgp_ai")


def test_target_base_urls_are_read_from_allowlisted_env(monkeypatch):
    monkeypatch.setenv("SGP_AI_LOCAL_API_BASE_URL", "http://local-env.test")
    monkeypatch.setenv("SGP_AI_BACKEND_API_BASE_URL", "https://backend-env.test")

    assert app_module._get_api_base("local") == "http://local-env.test"
    assert app_module._get_api_base("backend") == "https://backend-env.test"


def test_status_reports_installed_assistant(monkeypatch):
    _set_env(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith("http://backend.test/")
        assert request.headers["X-Api-Key"] == "backend-key"
        if request.url.path == "/assistants":
            return httpx.Response(
                200,
                json=[{"assistant_id": "sgp_ai", "display_name": "SGP AI"}],
                headers={"X-Request-Id": "req-1"},
            )
        if request.url.path == "/assistants/sgp_ai/documents":
            assert request.url.params["limit"] == "1"
            return httpx.Response(200, json=[{"document_id": "doc-1", "title": "Doc"}])
        raise AssertionError(f"unexpected request: {request.url}")

    _patch_async_client(monkeypatch, handler)
    with TestClient(app_module.app) as client:
        response = client.get("/sgp-ai-tester/api/status")
    assert response.status_code == 200
    assert response.json()["installed"] is True
    assert response.json()["assistant_id"] == "sgp_ai"
    assert response.json()["mode"] == "backend"
    assert response.json()["corpus_ready"] is True
    assert response.json()["document_probe_count"] == 1
    assert response.headers["X-Request-Id"] == "req-1"


def test_status_checks_backend_mode_with_backend_key(monkeypatch):
    _set_env(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith("http://backend.test/")
        assert request.headers["X-Api-Key"] == "backend-key"
        if request.url.path == "/assistants":
            return httpx.Response(200, json=[{"assistant_id": "sgp_ai", "display_name": "SGP AI"}])
        if request.url.path == "/assistants/sgp_ai/documents":
            return httpx.Response(200, json=[{"document_id": "doc-1"}])
        raise AssertionError(f"unexpected request: {request.url}")

    _patch_async_client(monkeypatch, handler)
    with TestClient(app_module.app) as client:
        response = client.get("/sgp-ai-tester/api/status?mode=backend")
    assert response.status_code == 200
    assert response.json()["installed"] is True
    assert response.json()["mode"] == "backend"
    assert response.json()["corpus_ready"] is True


def test_status_reports_installed_assistant_with_missing_corpus(monkeypatch):
    _set_env(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith("http://backend.test/")
        if request.url.path == "/assistants":
            return httpx.Response(200, json=[{"assistant_id": "sgp_ai", "display_name": "SGP AI"}])
        if request.url.path == "/assistants/sgp_ai/documents":
            return httpx.Response(200, json=[])
        raise AssertionError(f"unexpected request: {request.url}")

    _patch_async_client(monkeypatch, handler)
    with TestClient(app_module.app) as client:
        response = client.get("/sgp-ai-tester/api/status")
    assert response.status_code == 200
    assert response.json()["installed"] is True
    assert response.json()["corpus_ready"] is False
    assert response.json()["document_probe_count"] == 0


def test_status_reports_missing_assistant(monkeypatch):
    _set_env(monkeypatch)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"assistant_id": "sample"}])

    _patch_async_client(monkeypatch, handler)
    with TestClient(app_module.app) as client:
        response = client.get("/sgp-ai-tester/api/status")
    assert response.status_code == 200
    assert response.json()["installed"] is False


def test_retrieve_proxies_debug_payload(monkeypatch):
    _set_env(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/assistants/sgp_ai/documents":
            return httpx.Response(
                200,
                json=[{"document_id": "doc-1", "title": "Doc", "year": 2024, "language": "en", "url": "https://example.org", "summary": "Summary"}],
            )
        assert request.url.path == "/assistants/sgp_ai/debug/retrieve"
        assert request.url.params["query"] == "coastal erosion"
        assert request.url.params["limit"] == "8"
        return httpx.Response(
            200,
            json={
                "assistant_id": "sgp_ai",
                "query": "coastal erosion",
                "documents": [{"title": "Doc", "year": 2024, "language": "en", "url": "https://example.org", "summary": "Summary"}],
                "chunks": [{"title": "Doc", "content": "Evidence", "page_start": 1, "page_end": 1}],
                "debug": {},
            },
        )

    _patch_async_client(monkeypatch, handler)
    with TestClient(app_module.app) as client:
        response = client.get("/sgp-ai-tester/api/retrieve?mode=local&query=coastal%20erosion&limit=8")
    assert response.status_code == 200
    assert response.json()["chunks"][0]["content"] == "Evidence"


def test_model_streams_ndjson(monkeypatch):
    _set_env(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/assistants/sgp_ai/documents":
            return httpx.Response(
                200,
                json=[{"document_id": "doc-1", "title": "Doc", "year": 2024, "language": "en", "url": "https://example.org", "summary": "Summary"}],
            )
        assert request.url.path == "/assistants/sgp_ai/model"
        assert json.loads(request.content.decode("utf-8")) == [{"role": "human", "content": "hello"}]
        return httpx.Response(
            200,
            content=b'{"role":"assistant","content":"Hello","graph":null}\n',
            headers={"content-type": "application/x-ndjson", "X-Request-Id": "req-2"},
        )

    _patch_async_client(monkeypatch, handler)
    with TestClient(app_module.app) as client:
        response = client.post(
            "/sgp-ai-tester/api/model",
            json={"target_mode": "backend", "messages": [{"role": "human", "content": "hello"}]},
        )
    assert response.status_code == 200
    assert response.headers["X-Request-Id"] == "req-2"
    assert response.text.strip() == '{"role":"assistant","content":"Hello","graph":null}'


def test_model_blocks_missing_corpus(monkeypatch):
    _set_env(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith("http://backend.test/")
        assert request.url.path == "/assistants/sgp_ai/documents"
        return httpx.Response(200, json=[])

    _patch_async_client(monkeypatch, handler)
    with TestClient(app_module.app) as client:
        response = client.post(
            "/sgp-ai-tester/api/model",
            json={"messages": [{"role": "human", "content": "hello"}]},
        )
    assert response.status_code == 409
    assert "corpus is not imported" in response.json()["detail"]


def test_model_requires_last_human_message(monkeypatch):
    _set_env(monkeypatch)
    with TestClient(app_module.app) as client:
        response = client.post(
            "/sgp-ai-tester/api/model",
            json={"messages": [{"role": "assistant", "content": "hello"}]},
        )
    assert response.status_code == 400
