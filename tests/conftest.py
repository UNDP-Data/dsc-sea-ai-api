"""
Shared pytest fixtures and setup for FastAPI application tests.
"""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def test_client():
    """
    Test client fixture using a context manager to trigger a lifespan.

    See https://fastapi.tiangolo.com/advanced/testing-events/.
    """
    with TestClient(app) as client:
        yield client
