"""
Basic tests to ensure API documentation pages are accessible.
"""


def test_read_homepage(test_client):
    """
    Test if the homepage is accessible (even without authentication).
    """
    # disable authentication
    test_client.headers.pop("X-Api-Key", None)
    response = test_client.get("/")
    assert response.status_code == 200
    assert "Swagger UI" in response.text
    assert "ReDoc" in response.text
    assert "Changelog" in response.text


def test_read_docs(test_client):
    """
    Test if Swagger UI documentation is accessible.
    """
    response = test_client.get("/docs")
    assert response.status_code == 200
    assert "Energy Academy AI API - Swagger UI" in response.text


def test_read_redoc(test_client):
    """
    Test if API changelog is accessible.
    """
    response = test_client.get("/redoc")
    assert response.status_code == 200
    assert "Energy Academy AI API - ReDoc" in response.text


def test_read_changelog(test_client):
    """
    Test if ReDoc documentation is accessible (even without authentication).
    """
    # disable authentication
    test_client.headers.pop("X-Api-Key", None)
    response = test_client.get("/changelog")
    assert response.status_code == 200
    assert "v0.1.0-beta (2025-05-13)" in response.text
    assert "© United Nations Development Programme" in response.text
