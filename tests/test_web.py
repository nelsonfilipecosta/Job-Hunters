"""Tests for the web application."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from job_hunters.config import ConfigError
from job_hunters.web import app


def test_health_returns_ok() -> None:
    """The `/health` endpoint responds 200 with a small ok body."""
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"Status": "Ok"}


def test_startup_validates_the_real_config() -> None:
    """Starting the app loads this repository's config without error."""
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


def test_startup_refuses_a_broken_config(monkeypatch) -> None:
    """A broken config stops the server from starting with a silent misconfiguration."""

    def _raise() -> None:
        raise ConfigError("`system_config.yaml` is invalid:\n  timezone: nope")

    monkeypatch.setattr("job_hunters.web.load_all", _raise)

    with pytest.raises(ConfigError, match="timezone"), TestClient(app):
        pass  # pragma: no cover -- startup raises before the body runs


def test_unknown_routes_404() -> None:
    """Only the routes we declared exist."""
    with TestClient(app) as client:
        assert client.get("/not-a-route").status_code == 404
