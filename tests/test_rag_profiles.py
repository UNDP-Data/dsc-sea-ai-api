"""Tests for reusable multi-topic RAG profile configuration."""

import pytest

from src import database, genai
from src.entities import Message
from src.rag_system import RagProfileError, get_profile, list_profiles, load_profile


def test_sea_profile_loads_with_default_tables():
    profile = get_profile("sea")
    assert profile.assistant_id == "sea"
    assert profile.is_default
    assert profile.table_names == {
        "chunks": "chunks",
        "documents": "documents",
        "sources": "sources",
    }
    assert profile.prompt_key("draft_answer") == "draft_answer"


def test_sample_profile_uses_isolated_table_namespace():
    profile = get_profile("sample")
    assert profile.assistant_id == "sample"
    assert not profile.is_default
    assert profile.table_name("chunks") == "sample_chunks"
    assert profile.table_name("documents") == "sample_documents"
    assert profile.table_name("sources") == "sample_sources"
    assert profile.prompt_text("draft_answer")


def test_missing_profile_raises_clear_error():
    with pytest.raises(RagProfileError, match="does not exist"):
        load_profile("missing-profile")


def test_list_profiles_includes_sea_and_sample():
    assistant_ids = {profile.assistant_id for profile in list_profiles()}
    assert {"sea", "sample"}.issubset(assistant_ids)


def test_database_client_uses_profile_table_names():
    profile = get_profile("sample")
    client = database.Client(connection=object(), profile=profile)
    assert client.table_name("chunks") == "sample_chunks"
    assert client.table_name("documents") == "sample_documents"
    assert client.table_name("sources") == "sample_sources"
    assert client.table_name("nodes") == "nodes"
    assert not client.enable_sea_current_data_policy


def test_sgp_profile_scope_allows_named_innovation_library_programmes():
    profile = get_profile("sgp_ai")
    decision = genai.assess_profile_scope(
        [
            Message(
                role="human",
                content="What does COMDEKS Phase 4 emphasize about societies in harmony with nature?",
            )
        ],
        profile,
    )

    assert decision.allowed is True
