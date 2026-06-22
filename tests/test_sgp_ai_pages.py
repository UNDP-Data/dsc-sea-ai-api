"""Tests for the public SGP AI GitHub Pages proxy."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

import main as app_module


ALLOWED_ORIGIN = "https://ben-keller.github.io"
ORG_ALLOWED_ORIGIN = "https://undp-data.github.io"


class FakeTable:
    async def count_rows(self):
        return 1433


class FakeClient:
    def table_name(self, logical_name: str) -> str:
        return f"sgp_ai_{logical_name}"

    async def open_optional_table(self, _name: str):
        return FakeTable()

    async def retrieve_chunks(self, query: str, *, limit: int, debug=None):
        assert query == "coastal erosion"
        assert limit == 2
        return [FakeDump({"content": "Evidence"})], [FakeDump({"title": "Doc"})]


class FakeDump:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return self.payload


def test_pages_status_requires_allowed_origin(monkeypatch):
    @asynccontextmanager
    async def fake_profile_client(_profile):
        yield FakeClient()

    monkeypatch.setattr(app_module, "_profile_client", fake_profile_client)

    with TestClient(app_module.app) as client:
        response = client.get(
            "/pages/sgp-ai/status",
            headers={"Origin": ALLOWED_ORIGIN},
        )
        blocked = client.get(
            "/pages/sgp-ai/status",
            headers={"Origin": "https://example.org"},
        )

    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == ALLOWED_ORIGIN
    assert response.json()["corpus_ready"] is True
    assert response.json()["document_count"] == 1433
    assert blocked.status_code == 403


def test_pages_status_rejects_missing_origin(monkeypatch):
    @asynccontextmanager
    async def fake_profile_client(_profile):
        yield FakeClient()

    monkeypatch.setattr(app_module, "_profile_client", fake_profile_client)

    with TestClient(app_module.app) as client:
        response = client.get("/pages/sgp-ai/status")

    assert response.status_code == 403


def test_pages_retrieve_returns_documents_and_chunks(monkeypatch):
    @asynccontextmanager
    async def fake_profile_client(_profile):
        yield FakeClient()

    monkeypatch.setattr(app_module, "_profile_client", fake_profile_client)

    with TestClient(app_module.app) as client:
        response = client.get(
            "/pages/sgp-ai/debug/retrieve?query=coastal%20erosion&limit=2",
            headers={"Origin": ALLOWED_ORIGIN},
        )

    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == ALLOWED_ORIGIN
    assert response.json()["documents"] == [{"title": "Doc"}]
    assert response.json()["chunks"] == [{"content": "Evidence"}]


def test_pages_preflight_allows_pages_origin():
    with TestClient(app_module.app) as client:
        response = client.options(
            "/pages/sgp-ai/model",
            headers={
                "Origin": ALLOWED_ORIGIN,
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.status_code == 204
    assert response.headers["Access-Control-Allow-Origin"] == ALLOWED_ORIGIN
    assert "POST" in response.headers["Access-Control-Allow-Methods"]


def test_pages_preflight_allows_undp_data_pages_origin():
    with TestClient(app_module.app) as client:
        response = client.options(
            "/pages/sgp-ai/model",
            headers={
                "Origin": ORG_ALLOWED_ORIGIN,
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.status_code == 204
    assert response.headers["Access-Control-Allow-Origin"] == ORG_ALLOWED_ORIGIN


def test_pages_model_streams_without_browser_api_key(monkeypatch):
    async def fake_ask_assistant_model(_request, assistant_id, messages):
        assert assistant_id == "sgp_ai"
        assert messages[-1].content == "hello"
        return StreamingResponse(
            iter([b'{"role":"assistant","content":"Hello","graph":null}\n']),
            media_type="application/x-ndjson",
        )

    monkeypatch.setattr(app_module, "ask_assistant_model", fake_ask_assistant_model)

    with TestClient(app_module.app) as client:
        response = client.post(
            "/pages/sgp-ai/model",
            headers={"Origin": ALLOWED_ORIGIN},
            json=[{"role": "human", "content": "hello"}],
        )

    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == ALLOWED_ORIGIN
    assert response.text.strip() == '{"role":"assistant","content":"Hello","graph":null}'
