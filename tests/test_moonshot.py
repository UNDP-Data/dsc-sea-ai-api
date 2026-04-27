"""Tests for the local Moonshot FastAPI mirror."""

from __future__ import annotations

from fastapi.testclient import TestClient

try:
    from api.main import app
    from api.src import moonshot
except ModuleNotFoundError:
    from main import app
    from src import moonshot

client = TestClient(app)


def clear_caches() -> None:
    moonshot.clear_runtime_state()


def test_health_reports_unconfigured_when_no_credentials(monkeypatch) -> None:
    monkeypatch.delenv("AZURE_OPENAI_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    clear_caches()

    response = client.get("/api/moonshot/health")

    assert response.status_code == 200
    assert response.json()["configured"] is False


def test_health_reports_unconfigured_for_placeholder_credentials(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_KEY", "your_azure_openai_key_here")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://your-resource-name.openai.azure.com/")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    clear_caches()

    response = client.get("/api/moonshot/health")

    assert response.status_code == 200
    assert response.json()["configured"] is False


def test_health_includes_cors_header_for_allowed_origin(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://undp-data.github.io")
    clear_caches()

    response = client.get(
        "/api/moonshot/health",
        headers={"Origin": "https://undp-data.github.io"},
    )

    assert response.status_code == 200
    assert_allowed_origin_headers(response)
    assert response.headers["x-moonshot-origin-match"] == "true"


def test_health_allows_published_origin_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    clear_caches()

    response = client.get(
        "/api/moonshot/health",
        headers={"Origin": "https://undp-data.github.io"},
    )

    assert response.status_code == 200
    assert_allowed_origin_headers(response)
    assert response.headers["x-moonshot-origin-match"] == "true"


def test_health_normalizes_allowed_origin_format(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "\"https://undp-data.github.io/\"")
    clear_caches()

    response = client.get(
        "/api/moonshot/health",
        headers={"Origin": "https://undp-data.github.io"},
    )

    assert response.status_code == 200
    assert_allowed_origin_headers(response)
    assert response.headers["x-moonshot-origin-match"] == "true"


def test_diagnostics_reports_runtime_cors_state(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://undp-data.github.io")
    clear_caches()

    response = client.get(
        "/api/moonshot/diagnostics",
        headers={"Origin": "https://undp-data.github.io"},
    )

    assert response.status_code == 200
    assert_allowed_origin_headers(response)
    payload = response.json()
    assert payload["corsVersion"] == moonshot.MOONSHOT_CORS_VERSION
    assert payload["receivedOrigin"] == "https://undp-data.github.io"
    assert payload["normalizedOrigin"] == "https://undp-data.github.io"
    assert payload["originMatch"] is True
    assert "https://undp-data.github.io" in payload["allowedOrigins"]


def test_diagnostics_reports_disallowed_origin_without_cors_allow_header(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://undp-data.github.io")
    clear_caches()

    response = client.get(
        "/api/moonshot/diagnostics",
        headers={"Origin": "https://example.com"},
    )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
    assert response.headers["x-moonshot-origin-match"] == "false"
    payload = response.json()
    assert payload["normalizedOrigin"] == "https://example.com"
    assert payload["originMatch"] is False


def test_parse_query_sanitizes_filters(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    clear_caches()
    monkeypatch.setattr(moonshot, "get_openai_client", lambda: object())
    monkeypatch.setattr(
        moonshot,
        "generate_parsed_filters",
        lambda **_: {
            "filters": {
                "countryCode": "KEN",
                "bureau": "rbap",
                "unknown": "value",
            },
            "unresolvedTerms": ["mini-grid entrepreneurship", "", "energy access"],
        },
    )

    response = client.post(
        "/api/moonshot/parse-query",
        headers={"Origin": "https://undp-data.github.io"},
        json={
            "query": "Projects in Kenya",
            "locale": "en",
            "filterCatalog": {
                "optionsByKey": {
                    "bureau": [{"value": "rbap", "label": "RBAP", "aliases": ["asia pacific"]}],
                    "countryCode": [{"value": "KEN", "label": "Kenya", "aliases": ["kenya"]}],
                }
            },
        },
    )

    assert response.status_code == 200
    assert_allowed_origin_headers(response)
    assert response.json() == {
        "filters": {"bureau": "rbap", "countryCode": "KEN"},
        "unresolvedTerms": ["mini-grid entrepreneurship", "energy access"],
    }


def test_parse_query_rejects_disallowed_origin(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://undp-data.github.io")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    clear_caches()

    response = client.post(
        "/api/moonshot/parse-query",
        headers={"X-Forwarded-For": "203.0.113.7"},
        json={
            "query": "Projects in Kenya",
            "locale": "en",
            "filterCatalog": {"optionsByKey": {}},
        },
    )

    assert response.status_code in {400, 403}
    assert "configured browser origins" in response_text(response)


def test_parse_query_applies_rate_limit(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://undp-data.github.io")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MOONSHOT_PARSE_RATE_LIMIT", "1")
    monkeypatch.setenv("MOONSHOT_RATE_LIMIT_WINDOW_SECONDS", "300")
    clear_caches()
    monkeypatch.setattr(moonshot, "get_openai_client", lambda: object())
    monkeypatch.setattr(
        moonshot,
        "generate_parsed_filters",
        lambda **_: {"filters": {"countryCode": "KEN"}, "unresolvedTerms": []},
    )

    payload = {
        "query": "Projects in Kenya",
        "locale": "en",
        "filterCatalog": {
            "optionsByKey": {
                "countryCode": [{"value": "KEN", "label": "Kenya", "aliases": ["kenya"]}],
            }
        },
    }
    headers = {
        "Origin": "https://undp-data.github.io",
        "X-Forwarded-For": "203.0.113.7",
    }

    first = client.post("/api/moonshot/parse-query", headers=headers, json=payload)
    second = client.post("/api/moonshot/parse-query", headers=headers, json=payload)

    assert first.status_code == 200
    assert second.status_code == 429
    assert "rate limit exceeded" in response_text(second)


def test_parse_query_options_returns_cors_headers(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://undp-data.github.io")
    clear_caches()

    response = client.options(
        "/api/moonshot/parse-query",
        headers={
            "Origin": "https://undp-data.github.io",
            "Access-Control-Request-Method": "POST",
            "X-Forwarded-For": "203.0.113.7",
        },
    )

    assert response.status_code in {200, 204}
    assert_allowed_origin_headers(response)


def test_parse_query_options_rejects_disallowed_origin(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://undp-data.github.io")
    clear_caches()

    response = client.options(
        "/api/moonshot/parse-query",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
            "X-Forwarded-For": "203.0.113.7",
        },
    )

    assert response.status_code in {400, 403}
    assert "access-control-allow-origin" not in response.headers
    assert response.headers["x-moonshot-origin-match"] == "false"


def test_project_synopsis_short_circuits_for_zero_projects(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    clear_caches()
    monkeypatch.setattr(moonshot, "get_openai_client", lambda: object())

    response = client.post(
        "/api/moonshot/project-synopsis",
        json={
            "query": "Projects in Kenya",
            "locale": "en",
            "filters": {"countryCode": "KEN"},
            "summaryMetrics": {
                "projectCount": 0,
                "countryCount": 0,
                "totalBudget": 0,
                "directBeneficiaries": 0,
                "vfBeneficiaries": 0,
                "nonVfBeneficiaries": 0,
                "cleanElectricityBeneficiaries": 0,
                "cleanCookingBeneficiaries": 0,
                "productiveUseBeneficiaries": 0,
                "policyProjectCount": 0,
                "topBeneficiaryCategories": [],
            },
            "projectContext": {
                "totalProjects": 0,
                "topProjects": [],
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["synopsis"] == "No matching projects are available for an AI-generated project overview."


def test_project_synopsis_returns_config_error_when_credentials_missing(monkeypatch) -> None:
    monkeypatch.delenv("AZURE_OPENAI_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    clear_caches()

    response = client.post(
        "/api/moonshot/project-synopsis",
        json={
            "query": "Projects in Kenya",
            "locale": "en",
            "filters": {"countryCode": "KEN"},
            "summaryMetrics": {
                "projectCount": 1,
                "countryCount": 1,
                "totalBudget": 10,
                "directBeneficiaries": 20,
                "vfBeneficiaries": 0,
                "nonVfBeneficiaries": 20,
                "cleanElectricityBeneficiaries": 10,
                "cleanCookingBeneficiaries": 0,
                "productiveUseBeneficiaries": 0,
                "policyProjectCount": 0,
                "topBeneficiaryCategories": [],
            },
            "projectContext": {
                "totalProjects": 1,
                "topProjects": [],
            },
        },
    )

    assert response.status_code == 503
    payload = response.json()
    message = (payload.get("error") or payload.get("detail") or "").lower()
    assert "credentials are not configured" in message


def response_text(response) -> str:
    payload = response.json()
    return ((payload.get("error") or payload.get("detail") or "")).lower()


def assert_allowed_origin_headers(response) -> None:
    assert response.headers["access-control-allow-origin"] == "https://undp-data.github.io"
    assert response.headers["x-moonshot-cors-version"] == moonshot.MOONSHOT_CORS_VERSION
