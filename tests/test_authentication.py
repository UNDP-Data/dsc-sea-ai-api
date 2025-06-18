"""
Authentication tests to ensure API access is limited.
"""

import pytest

from utils import get_response

endpoints = [
    ("/nodes", "GET", None),
    ("/nodes/solar%20energy", "GET", None),
    ("/graph", "GET", {"query": "solar energy"}),
    ("/model", "POST", [{"role": "human", "content": "Hi there."}]),
]


@pytest.mark.parametrize("endpoint,method,payload", endpoints)
def test_authentication_valid_key(
    test_client, endpoint: str, method: str, payload: dict | list | None
):
    """
    Test if authentication with a valid key works as expected.
    """
    response = get_response(test_client, endpoint, method, payload)
    assert response.status_code == 200


@pytest.mark.parametrize("endpoint,method,payload", endpoints)
def test_authentication_no_key(
    test_client, endpoint: str, method: str, payload: dict | list | None
):
    """
    Test if a missing key raises an exception.
    """
    test_client.headers.pop("X-Api-Key", None)
    response = get_response(test_client, endpoint, method, payload)
    assert response.status_code == 403


@pytest.mark.parametrize("endpoint,method,payload", endpoints)
def test_authentication_invalid_key(
    test_client, endpoint: str, method: str, payload: dict | list | None
):
    """
    Test if an invalid key raises an exception.
    """
    test_client.headers["X-Api-Key"] = "1234"
    response = get_response(test_client, endpoint, method, payload)
    assert response.status_code == 401
