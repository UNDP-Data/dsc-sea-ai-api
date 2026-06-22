"""Validation helpers for copyable assistant kits."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any

import yaml

from .profiles import DEFAULT_ASSISTANT_ID, LOGICAL_TABLES, RagProfile, RagProfileError, load_profile_from_mapping

ASSISTANT_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,62}$")
KIT_PROFILE_FILE = "assistant.yaml"
KIT_MANIFEST_FILE = Path("corpus") / "manifest.yaml"
KIT_EVAL_FILE = Path("eval") / "questions.yaml"


class AssistantKitError(ValueError):
    """Raised when a copyable assistant kit is invalid."""


@dataclass(frozen=True)
class AssistantKit:
    """Validated assistant kit paths and profile."""

    path: Path
    profile_path: Path
    manifest_path: Path | None
    eval_path: Path | None
    profile: RagProfile


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    if not path.exists():
        raise AssistantKitError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise AssistantKitError(f"YAML file must contain a mapping: {path}")
    return payload


def validate_assistant_id(assistant_id: str) -> str:
    """Validate assistant ids for file names, table prefixes, and routes."""
    clean = (assistant_id or "").strip()
    if not ASSISTANT_ID_RE.fullmatch(clean):
        raise AssistantKitError(
            "assistant_id must start with a lowercase letter and contain only "
            "lowercase letters, digits, and underscores."
        )
    return clean


def default_table_names(assistant_id: str) -> dict[str, str]:
    """Return the standard table namespace for an assistant id."""
    assistant_id = validate_assistant_id(assistant_id)
    if assistant_id == DEFAULT_ASSISTANT_ID:
        return {"chunks": "chunks", "documents": "documents", "sources": "sources"}
    return {logical: f"{assistant_id}_{logical}" for logical in LOGICAL_TABLES}


def _validate_standard_table_names(profile: RagProfile) -> None:
    expected = default_table_names(profile.assistant_id)
    if profile.table_names != expected:
        raise AssistantKitError(
            "Assistant profile table names must use the standard namespace "
            f"for {profile.assistant_id!r}: expected {expected}, got {profile.table_names}."
        )


def load_profile_file(path: Path, *, enforce_standard_tables: bool = True) -> RagProfile:
    """Load and validate a profile YAML file from an arbitrary path."""
    raw = load_yaml_file(path)
    assistant_id = validate_assistant_id(str(raw.get("assistant_id") or ""))
    profile = load_profile_from_mapping(raw, assistant_id=assistant_id, source=str(path))
    if enforce_standard_tables:
        _validate_standard_table_names(profile)
    return profile


def validate_manifest_mapping(payload: dict[str, Any], *, expected_assistant_id: str | None = None) -> list[str]:
    """Validate the high-level corpus manifest shape used by assistant kits."""
    errors: list[str] = []
    assistant_id = payload.get("assistant_id")
    if expected_assistant_id and assistant_id not in {None, expected_assistant_id}:
        errors.append(
            f"manifest assistant_id must be {expected_assistant_id!r} when present; got {assistant_id!r}."
        )
    if assistant_id is not None:
        try:
            validate_assistant_id(str(assistant_id))
        except AssistantKitError as error:
            errors.append(f"manifest assistant_id: {error}")

    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        errors.append("manifest must include a non-empty sources list.")
    else:
        seen_sources: set[str] = set()
        for index, source in enumerate(sources):
            prefix = f"sources[{index}]"
            if not isinstance(source, dict):
                errors.append(f"{prefix} must be a mapping.")
                continue
            for field in ("source_id", "name", "organization"):
                if not isinstance(source.get(field), str) or not source[field].strip():
                    errors.append(f"{prefix}.{field} is required.")
            if isinstance(source.get("source_id"), str):
                seen_sources.add(source["source_id"])

    documents = payload.get("documents")
    if not isinstance(documents, list) or not documents:
        errors.append("manifest must include a non-empty documents list.")
    else:
        source_ids = {
            source.get("source_id")
            for source in sources or []
            if isinstance(source, dict) and isinstance(source.get("source_id"), str)
        }
        for index, document in enumerate(documents):
            prefix = f"documents[{index}]"
            if not isinstance(document, dict):
                errors.append(f"{prefix} must be a mapping.")
                continue
            for field in ("source_id", "title", "url", "language", "summary"):
                if not isinstance(document.get(field), str) or not document[field].strip():
                    errors.append(f"{prefix}.{field} is required.")
            if document.get("source_id") not in source_ids:
                errors.append(f"{prefix}.source_id must match a source in manifest.sources.")
            try:
                year = int(document.get("year") or 0)
            except (TypeError, ValueError):
                year = 0
            if year <= 0:
                errors.append(f"{prefix}.year must be a positive integer.")
            has_content = isinstance(document.get("content"), str) and document["content"].strip()
            chunks = document.get("chunks")
            if chunks is not None:
                if not isinstance(chunks, list) or not chunks:
                    errors.append(f"{prefix}.chunks must be a non-empty list when provided.")
                else:
                    for chunk_index, chunk in enumerate(chunks):
                        chunk_prefix = f"{prefix}.chunks[{chunk_index}]"
                        if not isinstance(chunk, dict):
                            errors.append(f"{chunk_prefix} must be a mapping.")
                        elif not isinstance(chunk.get("content"), str) or not chunk["content"].strip():
                            errors.append(f"{chunk_prefix}.content is required.")
            elif not has_content:
                errors.append(f"{prefix} must include either content or non-empty chunks.")
    return errors


def validate_manifest_file(path: Path, *, expected_assistant_id: str | None = None) -> dict[str, Any]:
    """Load and validate a corpus manifest file."""
    payload = load_yaml_file(path)
    errors = validate_manifest_mapping(payload, expected_assistant_id=expected_assistant_id)
    if errors:
        raise AssistantKitError("Invalid corpus manifest:\n- " + "\n- ".join(errors))
    return payload


def validate_kit(kit_path: Path) -> AssistantKit:
    """Validate a copyable assistant kit directory."""
    path = kit_path.expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise AssistantKitError(f"Assistant kit directory does not exist: {path}")
    profile_path = path / KIT_PROFILE_FILE
    profile = load_profile_file(profile_path)
    manifest_path = path / KIT_MANIFEST_FILE
    eval_path = path / KIT_EVAL_FILE
    if manifest_path.exists():
        validate_manifest_file(manifest_path, expected_assistant_id=profile.assistant_id)
    else:
        manifest_path = None
    return AssistantKit(
        path=path,
        profile_path=profile_path,
        manifest_path=manifest_path,
        eval_path=eval_path if eval_path.exists() else None,
        profile=profile,
    )
