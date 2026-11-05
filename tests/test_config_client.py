"""Phase 4: Groq provider preset and the hardened OpenAI-compatible client (mocked HTTP)."""

from __future__ import annotations

import json

import httpx
import pytest

from restaurant_recommender.config import Settings
from restaurant_recommender.llm.client import OpenAICompatibleClient, build_client, supports_strict_schema
from restaurant_recommender.llm.errors import LLMParseError, LLMRequestError, LLMResponseError, LLMTimeoutError

SECRET = "gsk_test_secret_value_123"


def make_settings(**env) -> Settings:
    return Settings(_env_file=None, **env)


# --------------------------------------------------------------------------- config


def test_groq_is_the_default_provider():
    s = make_settings(LLM_API_KEY="")
    assert s.llm_provider == "groq"
    assert s.resolved_base_url == "https://api.groq.com/openai/v1"
    assert s.resolved_model == "openai/gpt-oss-120b"
    assert s.resolved_judge_model == "openai/gpt-oss-20b"
    assert not s.llm_enabled


def test_groq_api_key_alias_enables_llm_and_stays_secret():
    s = make_settings(GROQ_API_KEY=SECRET)
    assert s.llm_enabled
    assert SECRET not in repr(s) and SECRET not in str(s.model_dump())


def test_overrides_and_blank_values():
    s = make_settings(LLM_API_KEY=SECRET, LLM_MODEL="llama-3.1-8b-instant", LLM_BASE_URL="  ", JUDGE_MODEL="")
    assert s.resolved_model == "llama-3.1-8b-instant"
    assert s.resolved_base_url == "https://api.groq.com/openai/v1"  # blank -> preset
    assert s.resolved_judge_model == "openai/gpt-oss-20b"


def test_openai_and_keyless_custom_providers():
    assert make_settings(LLM_PROVIDER="openai", LLM_API_KEY="").resolved_model == "gpt-4o-mini"
    local = make_settings(LLM_PROVIDER="custom", LLM_API_KEY="", LLM_BASE_URL="http://localhost:11434/v1", LLM_MODEL="llama3.1")
    assert local.llm_enabled and local.resolved_model == "llama3.1"


def test_build_client_uses_preset(monkeypatch):
    client = build_client(make_settings(GROQ_API_KEY=SECRET))
    assert isinstance(client, OpenAICompatibleClient)
    assert str(client._client.base_url).startswith("https://api.groq.com/openai/v1")
    assert client.model == "openai/gpt-oss-120b" and client.response_format == "json_schema"
    assert client.reasoning_effort == "low"
    assert build_client(make_settings(LLM_API_KEY="")) is None


def test_strict_schema_support_by_model():
    assert supports_strict_schema("openai/gpt-oss-120b")
    assert not supports_strict_schema("llama-3.3-70b-versatile")


# --------------------------------------------------------------------------- client


def ok_response(request: httpx.Request, content: str = '{"summary": "s", "recommendations": []}') -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(
        200,
        headers={"x-ratelimit-remaining-tokens": "5400", "x-ratelimit-remaining-requests": "998",
                 "x-ratelimit-reset-tokens": "6.2s"},
        json={"id": "c1", "object": "chat.completion", "created": 0, "model": body["model"],
              "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
              "usage": {"prompt_tokens": 1200, "completion_tokens": 150, "total_tokens": 1350}},
    )


def make_client(handler, model="llama-3.3-70b-versatile", **kwargs):
    sleeps: list[float] = []
    client = OpenAICompatibleClient(
        api_key=SECRET, base_url="https://api.groq.com/openai/v1", model=model, timeout=5, temperature=0.3,
        reasoning_effort="low", http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleeps.append, **kwargs,
    )
    return client, sleeps


MESSAGES = [{"role": "system", "content": "Reply in JSON."}, {"role": "user", "content": "hi"}]
SCHEMA = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}


def test_success_returns_text_usage_and_rate_limit_headers():
    seen = []
    client, _ = make_client(lambda r: (seen.append(json.loads(r.content)), ok_response(r))[1])
    result = client.complete(MESSAGES, schema=SCHEMA)
    assert result.text.startswith("{") and result.prompt_tokens == 1200 and result.completion_tokens == 150
    assert result.rate_limit["x-ratelimit-remaining-tokens"] == "5400"
    sent = seen[0]
    assert sent["response_format"] == {"type": "json_object"}  # llama: JSON mode
    assert "reasoning_effort" not in sent and sent["max_tokens"] == 1200


def test_gpt_oss_uses_strict_schema_and_low_reasoning():
    seen = []
    client, _ = make_client(lambda r: (seen.append(json.loads(r.content)), ok_response(r))[1], model="openai/gpt-oss-120b")
    client.complete(MESSAGES, schema=SCHEMA)
    assert seen[0]["response_format"]["type"] == "json_schema"
    assert seen[0]["response_format"]["json_schema"]["strict"] is True
    assert seen[0]["reasoning_effort"] == "low"


def test_rejected_schema_degrades_to_json_object():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        if len(seen) == 1:
            return httpx.Response(400, json={"error": {"message": "response_format json_schema is not supported", "code": "invalid"}})
        return ok_response(request)

    client, _ = make_client(handler, model="openai/gpt-oss-20b")
    client.complete(MESSAGES, schema=SCHEMA)
    assert [s["response_format"]["type"] for s in seen] == ["json_schema", "json_object"]
    assert client.response_format == "json_object"


def test_short_429_is_retried_after_waiting():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "1"}, json={"error": {"message": "rate limit"}})
        return ok_response(request)

    client, sleeps = make_client(handler)
    assert client.complete(MESSAGES).text
    assert sleeps == [1.0] and len(calls) == 2


def test_long_429_falls_back_immediately():
    client, sleeps = make_client(lambda r: httpx.Response(429, headers={"retry-after": "30"}, json={"error": {"message": "x"}}))
    with pytest.raises(LLMRequestError) as info:
        client.complete(MESSAGES)
    assert info.value.kind == "rate_limit" and "30" in str(info.value) and sleeps == []


def test_5xx_is_retried_once_then_raised():
    calls = []
    client, sleeps = make_client(lambda r: (calls.append(1), httpx.Response(503, json={"error": {"message": "x"}}))[1])
    with pytest.raises(LLMRequestError) as info:
        client.complete(MESSAGES)
    assert info.value.kind == "http" and len(calls) == 2 and sleeps == [0.5]


def test_auth_error_never_leaks_the_key():
    client, _ = make_client(lambda r: httpx.Response(401, json={"error": {"message": "Invalid API Key"}}))
    with pytest.raises(LLMRequestError) as info:
        client.complete(MESSAGES)
    assert info.value.kind == "auth" and SECRET not in str(info.value)


def test_groq_json_validate_failed_becomes_parse_error_with_raw():
    body = {"error": {"message": "Failed to generate JSON", "code": "json_validate_failed", "failed_generation": "{oops"}}
    client, _ = make_client(lambda r: httpx.Response(400, json=body))
    with pytest.raises(LLMParseError) as info:
        client.complete(MESSAGES)
    assert info.value.raw == "{oops" and info.value.kind == "parse"


def test_empty_content_and_timeout():
    client, _ = make_client(lambda r: ok_response(r, content="  "))
    with pytest.raises(LLMResponseError):
        client.complete(MESSAGES)

    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    client, _ = make_client(timeout)
    with pytest.raises(LLMTimeoutError):
        client.complete(MESSAGES)
