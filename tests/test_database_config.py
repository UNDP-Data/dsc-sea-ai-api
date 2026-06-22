"""Tests for LanceDB connection configuration."""

from src import database
from src.rag_system import get_profile


def test_lancedb_uri_defaults_to_azure(monkeypatch):
    monkeypatch.delenv("LANCEDB_URI", raising=False)

    assert database.get_lancedb_uri() == "az://lancedb"


def test_lancedb_uri_accepts_local_path(monkeypatch, tmp_path):
    local_path = tmp_path / "lancedb"
    monkeypatch.setenv("LANCEDB_URI", str(local_path))

    assert database.get_lancedb_uri() == str(local_path)
    assert local_path.exists()


def test_lancedb_uri_accepts_explicit_uri(monkeypatch):
    monkeypatch.setenv("LANCEDB_URI", "s3://example/lancedb")

    assert database.get_lancedb_uri() == "s3://example/lancedb"


def test_lancedb_uri_uses_profile_storage_when_no_env_override(monkeypatch):
    monkeypatch.delenv("LANCEDB_URI", raising=False)
    profile = get_profile("sgp_ai")

    assert database.get_lancedb_uri(profile=profile) == "az://lancedb/sgp_ai"


def test_lancedb_uri_env_override_wins_over_profile_storage(monkeypatch, tmp_path):
    local_path = tmp_path / "local-profile-override"
    monkeypatch.setenv("LANCEDB_URI", str(local_path))
    profile = get_profile("sgp_ai")

    assert database.get_lancedb_uri(profile=profile) == str(local_path)


def test_fallback_knowledge_graph_supports_local_graph_endpoints():
    graph = database.build_fallback_knowledge_graph()

    assert "solar energy" in graph
    assert "climate adaptation strategies" in graph
    assert graph.number_of_edges() > 1
    assert graph.out_degree("solar energy") > 1
