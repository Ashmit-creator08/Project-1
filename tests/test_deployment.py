"""Deployment settings: metrics token, CORS, request-id header, JSON logs, packaging files."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from restaurant_recommender import __version__
from restaurant_recommender.api.main import create_app
from restaurant_recommender.logging_config import JsonFormatter
from tests.helpers import FakeLLM

ROOT = Path(__file__).resolve().parents[1]


def client_for(store, settings, **overrides):
    app = create_app(settings=settings.model_copy(update=overrides), store=store, llm_client=FakeLLM())
    return TestClient(app)


def test_metrics_token_required_when_configured(store, settings):
    from pydantic import SecretStr

    with client_for(store, settings, metrics_token=SecretStr("s3cret")) as client:
        assert client.get("/metrics").status_code == 401
        assert client.get("/metrics", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert client.get("/metrics", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_metrics_open_without_token(store, settings):
    with client_for(store, settings) as client:
        assert client.get("/metrics").status_code == 200


def test_cors_only_when_configured(store, settings):
    preflight = {"Origin": "https://app.example.com", "Access-Control-Request-Method": "POST"}
    with client_for(store, settings) as client:
        response = client.options("/api/v1/recommendations", headers=preflight)
        assert "access-control-allow-origin" not in response.headers
    with client_for(store, settings, cors_allow_origins="https://app.example.com") as client:
        response = client.options("/api/v1/recommendations", headers=preflight)
        assert response.headers["access-control-allow-origin"] == "https://app.example.com"


def test_request_id_header(store, settings):
    with client_for(store, settings) as client:
        response = client.post("/api/v1/recommendations", json={"location": "Indiranagar"},
                               headers={"X-Request-ID": "trace-1"})
        assert response.headers["X-Request-ID"] == "trace-1"
        generated = client.post("/api/v1/recommendations", json={"location": "Whitefield"})
        assert generated.headers["X-Request-ID"] == generated.json()["meta"]["request_id"]


def test_json_log_formatter():
    record = logging.LogRecord("restaurant_recommender", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "hello world" and payload["level"] == "INFO"


def test_version_is_consistent():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{__version__}"' in pyproject


@pytest.mark.parametrize("name", ["Dockerfile", "docker-compose.yml", ".dockerignore", "docker/entrypoint.sh",
                                  "requirements.lock", ".streamlit/config.toml", ".github/workflows/ci.yml"])
def test_deployment_files_exist(name):
    assert (ROOT / name).is_file(), name


def test_entrypoint_uses_lf_line_endings():
    assert b"\r\n" not in (ROOT / "docker" / "entrypoint.sh").read_bytes()


def test_dockerignore_keeps_secrets_out():
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignored and ".venv" in ignored and ".git" in ignored


def test_lockfile_pins_everything_and_has_no_local_paths():
    lines = [l for l in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    assert lines and all("==" in l for l in lines)
    assert not any(l.startswith("-e") or "file:" in l or "git+" in l for l in lines)
    names = {l.split("==")[0].lower() for l in lines}
    assert {"fastapi", "uvicorn", "streamlit", "openai", "pandas", "pyarrow"} <= names
