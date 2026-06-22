"""Profile loading for copyable multi-topic RAG assistants."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ASSISTANT_ID = "sea"
PROFILE_DIR = Path("config/rag_profiles")
LOGICAL_TABLES = ("chunks", "documents", "sources")


class RagProfileError(ValueError):
    """Raised when a RAG profile is missing or invalid."""


@dataclass(frozen=True)
class RagProfile:
    """Runtime configuration for one publication-backed assistant."""

    assistant_id: str
    display_name: str
    domain_description: str
    refusal_guidance: str
    tables: dict[str, str]
    prompts: dict[str, str] = field(default_factory=dict)
    prompt_refs: dict[str, str] = field(default_factory=dict)
    source_definitions: list[dict[str, Any]] = field(default_factory=list)
    tag_rules: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    retrieval_policy: dict[str, Any] = field(default_factory=dict)
    scope: dict[str, Any] = field(default_factory=dict)
    storage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def table_names(self) -> dict[str, str]:
        return {logical: self.tables[logical] for logical in LOGICAL_TABLES}

    @property
    def is_default(self) -> bool:
        return self.assistant_id == DEFAULT_ASSISTANT_ID

    def table_name(self, logical_name: str) -> str:
        return self.tables.get(logical_name, logical_name)

    def prompt_key(self, name: str) -> str | None:
        return self.prompt_refs.get(name)

    def prompt_text(self, name: str) -> str | None:
        return self.prompts.get(name)

    def current_data_policy(self) -> dict[str, Any]:
        policy = self.retrieval_policy.get("current_data")
        return policy if isinstance(policy, dict) else {}

    @property
    def lancedb_uri(self) -> str | None:
        uri = self.storage.get("lancedb_uri")
        return uri.strip() if isinstance(uri, str) and uri.strip() else None


def _ensure_string(value: Any, field_name: str, assistant_id: str | None = None) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    suffix = f" for profile {assistant_id!r}" if assistant_id else ""
    raise RagProfileError(f"Missing or invalid `{field_name}`{suffix}.")


def _ensure_mapping(value: Any, field_name: str, assistant_id: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise RagProfileError(f"Missing or invalid `{field_name}` for profile {assistant_id!r}.")


def _normalise_table_names(raw_tables: dict[str, Any], assistant_id: str) -> dict[str, str]:
    tables: dict[str, str] = {}
    for logical_name in LOGICAL_TABLES:
        tables[logical_name] = _ensure_string(
            raw_tables.get(f"{logical_name}_table") or raw_tables.get(logical_name),
            f"tables.{logical_name}",
            assistant_id,
        )
    return tables


def _normalise_rules(value: Any) -> dict[str, dict[str, list[str]]]:
    if not isinstance(value, dict):
        return {}
    rules: dict[str, dict[str, list[str]]] = {}
    for group, raw_group in value.items():
        if not isinstance(raw_group, dict):
            continue
        group_rules: dict[str, list[str]] = {}
        for tag, patterns in raw_group.items():
            if isinstance(patterns, str):
                clean_patterns = [patterns]
            elif isinstance(patterns, list):
                clean_patterns = [item for item in patterns if isinstance(item, str) and item.strip()]
            else:
                clean_patterns = []
            if clean_patterns:
                group_rules[str(tag)] = clean_patterns
        rules[str(group)] = group_rules
    return rules


def _profile_path(assistant_id: str) -> Path:
    assistant_id = assistant_id.strip()
    if not assistant_id:
        raise RagProfileError("Assistant id cannot be empty.")
    if "/" in assistant_id or ".." in assistant_id:
        raise RagProfileError(f"Invalid assistant id {assistant_id!r}.")
    return PROFILE_DIR / f"{assistant_id}.yaml"


def load_profile_from_mapping(
    raw: dict[str, Any],
    *,
    assistant_id: str | None = None,
    source: str = "profile mapping",
) -> RagProfile:
    """Validate and build a RAG profile from a YAML mapping."""
    if not isinstance(raw, dict):
        raise RagProfileError(f"RAG profile from {source} must be a mapping.")

    profile_id = _ensure_string(raw.get("assistant_id"), "assistant_id")
    if assistant_id is not None and profile_id != assistant_id:
        raise RagProfileError(
            f"RAG profile {source} declares assistant_id={profile_id!r}, expected {assistant_id!r}."
        )
    tables = _normalise_table_names(_ensure_mapping(raw.get("tables"), "tables", profile_id), profile_id)
    prompts = raw.get("prompts") if isinstance(raw.get("prompts"), dict) else {}
    prompt_refs = raw.get("prompt_refs") if isinstance(raw.get("prompt_refs"), dict) else {}
    required_prompt_names = {"draft_answer", "answer_with_publications", "suggest_ideas"}
    missing_prompts = sorted(
        name for name in required_prompt_names if name not in prompts and name not in prompt_refs
    )
    if missing_prompts:
        raise RagProfileError(
            f"RAG profile {profile_id!r} is missing prompt text or refs for: {', '.join(missing_prompts)}."
        )

    source_definitions = raw.get("source_definitions")
    if not isinstance(source_definitions, list):
        source_definitions = []

    return RagProfile(
        assistant_id=profile_id,
        display_name=_ensure_string(raw.get("display_name"), "display_name", profile_id),
        domain_description=_ensure_string(raw.get("domain_description"), "domain_description", profile_id),
        refusal_guidance=_ensure_string(raw.get("refusal_guidance"), "refusal_guidance", profile_id),
        tables=tables,
        prompts={str(key): str(value) for key, value in prompts.items() if isinstance(value, str)},
        prompt_refs={str(key): str(value) for key, value in prompt_refs.items() if isinstance(value, str)},
        source_definitions=[item for item in source_definitions if isinstance(item, dict)],
        tag_rules=_normalise_rules(raw.get("tag_rules")),
        retrieval_policy=raw.get("retrieval_policy") if isinstance(raw.get("retrieval_policy"), dict) else {},
        scope=raw.get("scope") if isinstance(raw.get("scope"), dict) else {},
        storage=raw.get("storage") if isinstance(raw.get("storage"), dict) else {},
        raw=raw,
    )


@lru_cache(maxsize=32)
def load_profile(assistant_id: str = DEFAULT_ASSISTANT_ID) -> RagProfile:
    """Load and validate one RAG profile from config/rag_profiles."""
    path = _profile_path(assistant_id)
    if not path.exists():
        raise RagProfileError(f"RAG profile {assistant_id!r} does not exist at {path}.")
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return load_profile_from_mapping(raw, assistant_id=assistant_id, source=str(path))


def get_profile(assistant_id: str = DEFAULT_ASSISTANT_ID) -> RagProfile:
    return load_profile(assistant_id)


def list_profiles() -> list[RagProfile]:
    if not PROFILE_DIR.exists():
        return []
    profiles: list[RagProfile] = []
    for path in sorted(PROFILE_DIR.glob("*.yaml")):
        try:
            profiles.append(load_profile(path.stem))
        except RagProfileError:
            continue
    return profiles
