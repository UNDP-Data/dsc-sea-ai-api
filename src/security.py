"""
Security routines for authenticating requests.
"""

import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(
    name="X-Api-Key",
    description="The API key for authentication.",
    auto_error=True,
)


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
    if api_key == os.environ["API_KEY"]:
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key."
    )
