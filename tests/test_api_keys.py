"""
Unit tests for per-caller API keys. These do not need the data store.
"""

import asyncio
import json

import pytest
from fastapi import HTTPException

from src.security import authenticate, configured_api_keys


def run(key):
    return asyncio.run(authenticate(key))


def test_single_legacy_key_still_works(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.setenv("API_KEY", "legacy-key")
    assert run("legacy-key") is True
    with pytest.raises(HTTPException) as error:
        run("other")
    assert error.value.status_code == 401


def test_each_caller_has_its_own_key(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("API_KEYS", json.dumps({"sgp-pipeline": "sgp-key", "academy": "academy-key"}))
    assert run("sgp-key") is True
    assert run("academy-key") is True
    with pytest.raises(HTTPException):
        run("legacy-key")


def test_legacy_key_kept_during_migration(monkeypatch):
    monkeypatch.setenv("API_KEY", "legacy-key")
    monkeypatch.setenv("API_KEYS", json.dumps({"sgp-pipeline": "sgp-key"}))
    assert set(configured_api_keys()) == {"sgp-pipeline", "default"}
    assert run("legacy-key") is True
    assert run("sgp-key") is True


def test_empty_values_are_ignored_and_nothing_configured_is_an_error(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("API_KEYS", json.dumps({"retired": ""}))
    with pytest.raises(HTTPException) as error:
        run("")
    assert error.value.status_code == 500


def test_malformed_setting_fails_closed(monkeypatch):
    monkeypatch.setenv("API_KEY", "legacy-key")
    monkeypatch.setenv("API_KEYS", "not json")
    with pytest.raises(HTTPException) as error:
        run("legacy-key")
    assert error.value.status_code == 500
