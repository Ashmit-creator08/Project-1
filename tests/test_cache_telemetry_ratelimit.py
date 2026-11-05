"""Phase 4: response cache, telemetry and rate limiting (service + API)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from restaurant_recommender.api.main import create_app
from restaurant_recommender.api.ratelimit import TokenBucketLimiter
from restaurant_recommender.llm.client import LLMResult
from restaurant_recommender.llm.errors import LLMTimeoutError
from restaurant_recommender.models.schemas import PreferenceRequest
from restaurant_recommender.services.cache import TTLCache, cache_key
from restaurant_recommender.services.recommend import RecommendationService
from restaurant_recommender.services.telemetry import Telemetry, percentile
from tests.helpers import FakeLLM, reverse_ranker


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# --------------------------------------------------------------------------- cache


def test_ttl_cache_expiry_and_lru():
    clock = Clock()
    cache = TTLCache(max_entries=2, ttl_seconds=10, clock=clock)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1  # a is now most recent
    cache.set("c", 3)  # evicts b (least recently used)
    assert cache.get("b") is None and cache.get("c") == 3
    clock.now += 11
    assert cache.get("a") is None and len(cache) == 1  # expired entry removed on read
    assert cache.hits == 2 and cache.misses == 2


def _key(prefs: PreferenceRequest, **overrides) -> str:
    args = {"data_version": "v1", "prompt_variant": "v1_baseline", "model": "llama"} | overrides
    return cache_key(prefs, **args)


def test_cache_key_normalizes_equivalent_requests():  # CF-07
    a = PreferenceRequest(location=" indiranagar ", cuisine="Italian, Thai", budget="Medium", min_rating=3.5,
                          additional_preferences="Family  friendly", top_n=5)
    b = PreferenceRequest(location="Indira Nagar", cuisine="thai,italian", budget="medium", min_rating=3.50,
                          additional_preferences="family friendly", top_n=5)
    assert _key(a) == _key(b)


def test_cache_key_changes_with_data_prompt_model_or_prefs():  # CF-06
    prefs = PreferenceRequest(location="Indiranagar", top_n=5)
    base = _key(prefs)
    assert base != _key(prefs, data_version="v2")
    assert base != _key(prefs, prompt_variant="v2_stricter_json")
    assert base != _key(prefs, model="other")
    assert base != _key(PreferenceRequest(location="Indiranagar", top_n=4))


def test_service_serves_repeat_queries_from_cache(store, settings):
    llm = FakeLLM()
    service = RecommendationService(store, llm, settings)
    prefs = PreferenceRequest(location="Indiranagar", cuisine="Italian", top_n=2)
    first = service.recommend(prefs, request_id="r1")
    second = service.recommend(prefs, request_id="r2")
    assert len(llm.calls) == 1
    assert not first.meta.cache_hit and second.meta.cache_hit
    assert second.meta.request_id == "r2"
    assert [r.source_id for r in second.recommendations] == [r.source_id for r in first.recommendations]
    assert service.telemetry.snapshot()["cache"]["hits"] == 1


def test_fallback_responses_are_not_cached(store, settings):
    llm = FakeLLM(LLMTimeoutError("slow"))
    service = RecommendationService(store, llm, settings)
    prefs = PreferenceRequest(location="Indiranagar", top_n=2)
    assert service.recommend(prefs).meta.fallback_used
    recovered = service.recommend(prefs)
    assert recovered.meta.llm_used and not recovered.meta.cache_hit and len(llm.calls) == 2


def test_cache_can_be_disabled(store, settings):
    llm = FakeLLM()
    service = RecommendationService(store, llm, settings.model_copy(update={"cache_ttl_seconds": 0}))
    prefs = PreferenceRequest(location="Indiranagar", top_n=2)
    service.recommend(prefs)
    service.recommend(prefs)
    assert service.cache is None and len(llm.calls) == 2


def test_usage_and_prompt_variant_reach_meta(store, settings):
    llm = FakeLLM(lambda m: LLMResult(text=reverse_ranker(m), prompt_tokens=900, completion_tokens=120,
                                      rate_limit={"x-ratelimit-remaining-tokens": "7000"}))
    service = RecommendationService(store, llm, settings, prompt_variant="v2_stricter_json")
    meta = service.recommend(PreferenceRequest(location="Indiranagar", top_n=2)).meta
    assert meta.usage == {"prompt_tokens": 900, "completion_tokens": 120}
    assert meta.prompt_variant == "v2_stricter_json" and meta.model == "fake-llm"
    snap = service.telemetry.snapshot()
    assert snap["llm"]["prompt_tokens"] == 900
    assert snap["llm"]["rate_limit_remaining"] == {"x-ratelimit-remaining-tokens": "7000"}


def test_unknown_prompt_variant_is_rejected(store, settings):
    with pytest.raises(ValueError, match="PROMPT_VARIANT"):
        RecommendationService(store, None, settings, prompt_variant="v9")


# --------------------------------------------------------------------------- telemetry


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([10, 20, 30, 40], 50) == 25.0
    assert percentile([1.0] * 19 + [100.0], 95) == pytest.approx(5.9, abs=0.1)


def test_telemetry_rates_and_reasons():
    t = Telemetry()
    t.record_request(timings_ms={"filter": 5, "llm": 900, "total": 910}, empty=False, cache_hit=False,
                     llm_used=True, fallback_kind=None, llm_calls=1, prompt_tokens=1000, completion_tokens=100)
    t.record_request(timings_ms={"filter": 6, "llm": 20000, "total": 20010}, empty=False, cache_hit=False,
                     llm_used=False, fallback_kind="timeout", llm_calls=1)
    t.record_request(timings_ms={"total": 1}, empty=False, cache_hit=True, llm_used=False, fallback_kind=None)
    t.record_request(timings_ms={"filter": 4, "total": 4}, empty=True, cache_hit=False, llm_used=False, fallback_kind=None)
    t.record_rate_limited()
    snap = t.snapshot()
    assert snap["requests"] == 4 and snap["rate_limited"] == 1 and snap["empty_results"] == 1
    assert snap["fallback"] == {"count": 1, "rate": 0.5, "reasons": {"timeout": 1}}
    assert snap["cache"]["hit_rate"] == 0.25
    assert snap["latency_ms"]["llm"]["samples"] == 2 and snap["latency_ms"]["total"]["samples"] == 3


# --------------------------------------------------------------------------- rate limiting


def test_token_bucket():
    clock = Clock()
    limiter = TokenBucketLimiter(per_minute=2, clock=clock)
    assert limiter.check("a") == (True, 0)
    assert limiter.check("a") == (True, 0)
    allowed, retry = limiter.check("a")
    assert not allowed and retry == 30
    assert limiter.check("b")[0]  # per client
    clock.now += 30
    assert limiter.check("a")[0]
    assert TokenBucketLimiter(per_minute=0).check("x") == (True, 0)


@pytest.fixture
def limited_client(store, settings):
    app = create_app(settings=settings.model_copy(update={"rate_limit_per_minute": 2}), store=store, llm_client=FakeLLM())
    with TestClient(app) as client:
        yield client


def test_api_rate_limit_metrics_and_health(limited_client):
    body = {"location": "Indiranagar", "top_n": 2}
    first = limited_client.post("/api/v1/recommendations", json=body)
    second = limited_client.post("/api/v1/recommendations", json=body)
    third = limited_client.post("/api/v1/recommendations", json=body)
    assert first.status_code == second.status_code == 200
    assert second.json()["meta"]["cache_hit"] is True
    assert third.status_code == 429 and int(third.headers["Retry-After"]) >= 1

    metrics = limited_client.get("/metrics").json()
    assert metrics["requests"] == 2 and metrics["rate_limited"] == 1
    assert metrics["cache"]["hits"] == 1 and metrics["cache"]["entries"] == 1
    assert "api_key" not in str(metrics).lower()

    health = limited_client.get("/health").json()
    assert health["llm_provider"] == "groq" and health["prompt_variant"] == "v2_stricter_json"
    assert health["rate_limit_per_minute"] == 2 and health["data_version"]
