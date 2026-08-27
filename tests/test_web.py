"""Tests for the web application's endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from job_hunters.web import app


def test_health_returns_ok() -> None:
    """The `/health` endpoint responds 200 with a small ok body."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"Status": "Ok"}
