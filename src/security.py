"""
Security routines for authenticating requests.
"""

import hmac
import json
import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(
    name="X-Api-Key",
    description="The API key for authentication.",
    auto_error=True,
)


def configured_api_keys() -> dict[str, str]:
    """
    Return the accepted API keys, keyed by caller label.

    `API_KEYS` holds a JSON object with one key per caller, for example
    `{"academy-frontend": "...", "sgp-pipeline": "..."}`, so each caller can be
    rotated or revoked on its own. The single `API_KEY` setting is still
    accepted as the caller labelled `default`.
    """
    keys: dict[str, str] = {}
    raw = os.getenv("API_KEYS", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        if not isinstance(parsed, dict):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="API keys are misconfigured on the server.",
            )
        keys.update(
            {str(label): value for label, value in parsed.items() if isinstance(value, str) and value}
        )
    legacy = os.getenv("API_KEY")
    if legacy:
        keys.setdefault("default", legacy)
    return keys


def matching_caller(api_key: str, keys: dict[str, str]) -> str | None:
    """Compare against every configured key so timing does not reveal which one matched."""
    match = None
    for label, expected in keys.items():
        if hmac.compare_digest(api_key.encode(), expected.encode()) and match is None:
            match = label
    return match


async def authenticate(api_key: str = Security(api_key_header)) -> bool:
    """
    Authenticate a request using an API Key header.

    Parameters
    ----------
    api_key : str
        API key as supplied by the client.

    Returns
    -------
    bool
        True if the authentication has been successful.

    Raises
    ------
    HTTPException
        If no key is provided or the key is invalid.
    """
    keys = configured_api_keys()
    if not keys:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key is not configured on the server.",
        )
    if matching_caller(api_key, keys) is not None:
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key."
    )


async def authenticate_sgp_project_access(
    api_key: str = Security(api_key_header),
) -> bool:
    """Authenticate the separate server-side credential for non-public SGP corpora."""
    expected_key = os.getenv("SGP_PROJECT_RAG_API_KEY")
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SGP project-corpus access is not configured on the server.",
        )
    if hmac.compare_digest(api_key, expected_key):
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid SGP project-corpus API key.",
    )
