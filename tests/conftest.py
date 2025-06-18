"""
Shared pytest fixtures and setup for FastAPI application tests.
"""

import os

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def test_client():
    """
    Test client fixture using a context manager to trigger a lifespan.

    See https://fastapi.tiangolo.com/advanced/testing-events/.
    """
    with TestClient(app, headers={"X-Api-Key": os.environ["API_KEY"]}) as client:
        yield client
