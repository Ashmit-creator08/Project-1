"""Eval set (docs/eval.md) against the real ingested dataset, with the offline mock LLM.

Uses the same harness code as `python -m restaurant_recommender.eval.run`, so a
green test run means the scorecard's hard metrics are green too. Skipped when
data/processed/restaurants.parquet has not been built.
"""

from __future__ import annotations

import time

import pytest

from restaurant_recommender.config import Settings, get_settings
from restaurant_recommender.data.filter import filter_restaurants
from restaurant_recommender.data.store import RestaurantStore
from restaurant_recommender.eval import run as harness
from restaurant_recommender.eval.mock import OfflineRanker
from restaurant_recommender.models.schemas import PreferenceRequest

DATA_PATH = get_settings().data_path
CASES = harness.load_cases(harness.DEFAULT_CASES)
pytestmark = pytest.mark.skipif(not DATA_PATH.exists(), reason="run `python -m restaurant_recommender.data.ingest` first")


@pytest.fixture(scope="module")
def real_store() -> RestaurantStore:
    return RestaurantStore.from_parquet(DATA_PATH)


@pytest.fixture(scope="module")
def eval_settings() -> Settings:
    return Settings(_env_file=None, LLM_API_KEY="", CACHE_TTL_SECONDS=0)


@pytest.fixture(scope="module")
def mock_report(real_store, eval_settings):
    report = harness.VariantReport("v1_baseline")
    throttle = harness.Throttle(report.telemetry, delay=0, enabled=False)
    harness.run_cases(CASES, real_store, eval_settings, "v1_baseline", OfflineRanker(), None, throttle, report, live=False)
    report.injections = harness.run_injections(real_store, eval_settings, "v1_baseline")
    return {o.case_id: o for o in report.outcomes}, report


@pytest.mark.parametrize("case_id", [c["id"] for c in CASES])
def test_eval_case(case_id, mock_report):
    outcomes, _ = mock_report
    outcome = outcomes[case_id]
    assert outcome.passed, outcome.failures


def test_hard_metrics_and_injection_suite(mock_report):
    _, report = mock_report
    for metric in ("id_valid", "constraint_ok", "fact_ok", "complete"):
        assert report.ratio(metric) == 1.0, metric
    assert report.cardinality == 1.0 and report.duplicate_free == 1.0
    ok_empty, n_empty = report.empty_path
    assert n_empty >= 2 and ok_empty == n_empty
    assert all(passed for _, passed, _ in report.injections), report.injections
    assert report.hard_pass


def test_fuzzy_matching_never_rescues_unknown_cities(real_store, eval_settings):
    for location in ["Delhi", "Atlantis", "Mumbai", "Chennai", "Hyderabad"]:
        result = filter_restaurants(real_store, PreferenceRequest(location=location), eval_settings.max_candidates)
        assert result.candidates == [], (location, result.notes)


def test_filter_latency_under_100ms(real_store, eval_settings):  # architecture §10 target
    prefs = PreferenceRequest(location="Bangalore", cuisine="Italian", budget="medium", min_rating=3.5)
    timings = []
    for _ in range(5):
        started = time.perf_counter()
        filter_restaurants(real_store, prefs, eval_settings.max_candidates)
        timings.append((time.perf_counter() - started) * 1000)
    assert min(timings) < 100, timings


def test_harness_cli_mock_ab_writes_scorecard(tmp_path):
    out = tmp_path / "scorecard.md"
    code = harness.main(["--llm", "mock", "--prompt", "all", "--latency", "2", "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert code == 0, text
    assert "## Hard metrics" in text and "## Prompt A/B" in text
    assert "| Constraint adherence | 100.0% | 100.0% | 100% |" in text
    assert "**Decision:** Ship" in text
