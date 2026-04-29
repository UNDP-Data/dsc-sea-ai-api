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


def test_allowed_origins_are_normalized(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "\"https://undp-data.github.io/\"")
    clear_caches()

    assert moonshot.get_allowed_origins() == ["https://undp-data.github.io"]


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


def test_parse_query_prefers_explicit_bureau_over_accidental_country(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    clear_caches()
    monkeypatch.setattr(moonshot, "get_openai_client", lambda: object())
    monkeypatch.setattr(
        moonshot,
        "generate_parsed_filters",
        lambda **_: {
            "filters": {
                "bureau": "all",
                "countryCode": "NER",
            },
            "unresolvedTerms": [],
        },
    )

    response = client.post(
        "/api/moonshot/parse-query",
        json={
            "query": "Clean cooking in RBAP",
            "locale": "en",
            "filterCatalog": {
                "optionsByKey": {
                    "bureau": [
                        {
                            "value": "RBAP",
                            "label": "RBAP",
                            "aliases": ["rbap", "asia pacific", "asia"],
                        },
                    ],
                    "countryCode": [
                        {"value": "NER", "label": "Niger", "aliases": ["niger", "ner"]},
                    ],
                }
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["filters"] == {"bureau": "RBAP"}


def test_parse_query_applies_explicit_beneficiary_subcategory(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    clear_caches()
    monkeypatch.setattr(moonshot, "get_openai_client", lambda: object())
    monkeypatch.setattr(
        moonshot,
        "generate_parsed_filters",
        lambda **_: {
            "filters": {
                "category": "all",
                "subCategory": "all",
            },
            "unresolvedTerms": [],
        },
    )

    response = client.post(
        "/api/moonshot/parse-query",
        json={
            "query": "Show clean cooking projects in Asia",
            "locale": "en",
            "filterCatalog": {
                "optionsByKey": {
                    "category": [
                        {"value": "Energy Access", "label": "Energy Access", "aliases": ["energy access"]},
                        {"value": "Energy Transition", "label": "Energy Transition", "aliases": ["energy transition"]},
                    ],
                    "subCategory": [
                        {
                            "value": "Clean Cooking",
                            "label": "Clean Cooking",
                            "aliases": ["clean cooking", "cooking"],
                        },
                    ],
                    "bureau": [
                        {"value": "RBAP", "label": "RBAP", "aliases": ["asia", "asia pacific", "rbap"]},
                    ],
                }
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["filters"] == {
        "bureau": "RBAP",
        "category": "Energy Access",
        "subCategory": "Clean Cooking",
    }


def test_parse_query_applies_semantic_aliases_from_catalog(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    clear_caches()
    monkeypatch.setattr(moonshot, "get_openai_client", lambda: object())
    monkeypatch.setattr(
        moonshot,
        "generate_parsed_filters",
        lambda **_: {
            "filters": {
                "category": "all",
                "subCategory": "all",
            },
            "unresolvedTerms": [],
        },
    )

    response = client.post(
        "/api/moonshot/parse-query",
        headers={"Origin": "https://undp-data.github.io"},
        json={
            "query": "show projects on e-mobility",
            "locale": "en",
            "filterCatalog": {
                "optionsByKey": {
                    "category": [
                        {"value": "Energy Access", "label": "Energy Access", "aliases": ["energy access"]},
                    ],
                    "subCategory": [
                        {
                            "value": "Transport",
                            "label": "Transport",
                            "aliases": ["transport", "e-mobility", "electric mobility"],
                        },
                    ],
                }
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["filters"] == {
        "category": "Energy Access",
        "subCategory": "Transport",
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


def test_project_synopsis_prompt_requires_named_project_examples(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    clear_caches()
    captured = {}
    monkeypatch.setattr(moonshot, "get_openai_client", lambda: object())

    def fake_generate_project_synopsis(**kwargs):
        captured.update(kwargs)
        return "Named project synopsis."

    monkeypatch.setattr(moonshot, "generate_project_synopsis", fake_generate_project_synopsis)

    response = client.post(
        "/api/moonshot/project-synopsis",
        json={
            "query": "Clean cooking in Asia",
            "locale": "en",
            "filters": {"bureau": "RBAP", "subCategory": "Clean Cooking"},
            "summaryMetrics": {
                "projectCount": 2,
                "countryCount": 2,
                "totalBudget": 1000,
                "directBeneficiaries": 300,
                "vfBeneficiaries": 100,
                "nonVfBeneficiaries": 200,
                "cleanElectricityBeneficiaries": 0,
                "cleanCookingBeneficiaries": 300,
                "productiveUseBeneficiaries": 0,
                "policyProjectCount": 1,
                "topBeneficiaryCategories": [{"category": "Clean Cooking", "value": 300}],
            },
            "projectContext": {
                "totalProjects": 2,
                "topProjects": [
                    {
                        "id": "project-1",
                        "title": "Clean Cooking Scale Up",
                        "countryName": "Cambodia",
                        "description": "Expands access to clean cooking technologies.",
                        "budget": 600,
                        "directBeneficiaries": 250,
                        "primaryOutputCategories": ["Energy Access"],
                    },
                    {
                        "id": "project-2",
                        "title": "Efficient Household Energy",
                        "countryName": "Nepal",
                        "description": "Supports household energy access and clean cooking.",
                        "budget": 400,
                        "directBeneficiaries": 50,
                        "primaryOutputCategories": ["Energy Access", "Policy"],
                    },
                ],
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["synopsis"] == "Named project synopsis."
    prompt = captured["system_prompt"]
    user_payload = captured["user_payload"]
    assert "Write the entire response in English." in prompt
    assert "Keep project titles and supplied country names exactly as provided" in prompt
    assert "Mention two to four specific top project titles exactly as supplied" in prompt
    assert "Do not repeat the deterministic summary totals" in prompt
    assert (
        "start the paragraph with a natural equivalent of: This subset of the UNDP energy portfolio"
    ) in prompt
    assert "Clean Cooking Scale Up" in [project["title"] for project in user_payload["projectContext"]["topProjects"]]
    assert user_payload["projectContext"]["topProjects"][0]["directBeneficiaries"] == 250


def test_project_synopsis_prompt_uses_global_opening_when_no_filters(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    clear_caches()

    captured: dict[str, object] = {}

    def fake_generate_project_synopsis(**kwargs):
        captured.update(kwargs)
        return "Global project synopsis."

    monkeypatch.setattr(moonshot, "get_openai_client", lambda: object())
    monkeypatch.setattr(moonshot, "generate_project_synopsis", fake_generate_project_synopsis)

    response = client.post(
        "/api/moonshot/project-synopsis",
        json={
            "query": "Generate an overview",
            "locale": "en",
            "filters": {
                "funding": "all",
                "genderMarker": "all",
                "category": "all",
                "subCategory": "all",
                "bureau": "all",
                "economy": "all",
                "hdiTier": "all",
                "specialGrouping": "all",
                "continentRegion": "all",
                "subRegion": "all",
                "sahel": "all",
                "crisis": "all",
                "countryCode": "all",
            },
            "summaryMetrics": {"projectCount": 1},
            "projectContext": {
                "totalProjects": 1,
                "topProjects": [
                    {
                        "id": "project-1",
                        "title": "Global Energy Project",
                        "countryName": "Kenya",
                        "description": "Energy project.",
                        "budget": 100,
                        "directBeneficiaries": 50,
                        "primaryOutputCategories": ["Energy Access"],
                    },
                ],
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["synopsis"] == "Global project synopsis."
    assert (
        "start the paragraph with a natural equivalent of: The UNDP energy portfolio"
    ) in captured["system_prompt"]


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
