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
    moonshot.get_settings.cache_clear()
    moonshot.get_openai_client.cache_clear()


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
    assert response.json() == {
        "filters": {"bureau": "rbap", "countryCode": "KEN"},
        "unresolvedTerms": ["mini-grid entrepreneurship", "energy access"],
    }


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
