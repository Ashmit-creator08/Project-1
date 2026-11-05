"""Headless Streamlit UI test (streamlit.testing.v1.AppTest): no browser, no LLM call."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from restaurant_recommender.config import get_settings  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "ui" / "app.py")


@pytest.fixture
def app_env(monkeypatch):
    monkeypatch.setenv("AUTO_INGEST", "0")  # never download in tests
    monkeypatch.delenv("RECOMMENDER_API_URL", raising=False)


def test_ui_renders_form_or_clear_data_error(app_env):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Where should we eat?"
    if get_settings().data_path.exists():
        labels = [s.label for s in at.selectbox]
        assert labels == ["Location", "Cuisine"]
        assert at.selectbox[0].value == "Bangalore"
        assert any("restaurants" in c.value for c in at.caption)
    else:  # e.g. CI: the dataset is gitignored
        assert at.error and "ingest" in at.error[0].value


def test_requirements_txt_installs_the_lockfile():
    text = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(encoding="utf-8")
    assert "-r requirements.lock" in text
