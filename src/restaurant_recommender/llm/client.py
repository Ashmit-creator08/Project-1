"""Provider-agnostic chat client for OpenAI-compatible APIs (Groq by default, OpenAI, Ollama...).

Phase 4 hardening:

* Groq preset (see `config.PROVIDER_PRESETS`).
* Structured output: strict `json_schema` on models that support it, else
  `json_object`, degrading automatically when a provider rejects a format.
* Controlled retries: one retry for 5xx/connection errors; a 429 is retried
  once only if `retry-after` is short, otherwise we fall back immediately.
* Groq `json_validate_failed` -> `LLMParseError` carrying `failed_generation`.
* Token usage and `x-ratelimit-remaining-*` headers returned for telemetry.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import openai
from openai import OpenAI

from restaurant_recommender.config import Settings
from restaurant_recommender.llm.errors import (
    LLMParseError,
    LLMRequestError,
    LLMResponseError,
    LLMTimeoutError,
)

log = logging.getLogger(__name__)

Messages = list[dict[str, str]]

# Model-id prefixes with strict json_schema support (Groq docs, Nov 2026; OpenAI GPT-4o+).
STRICT_SCHEMA_PREFIXES = (
    "openai/gpt-oss",
    "qwen/qwen3",
    "meta-models/muse",
    "gpt-4o",
    "gpt-4.1",
    "gpt-5",
)
REASONING_PREFIXES = ("openai/gpt-oss",)
_RATE_LIMIT_HEADERS = (
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-tokens",
)


@dataclass
class LLMResult:
    text: str
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    rate_limit: dict[str, str] = field(default_factory=dict)


class ChatClient(Protocol):
    """Anything that turns chat messages into a completion (str or LLMResult)."""

    model: str

    def complete(self, messages: Messages, *, schema: dict | None = None) -> LLMResult | str: ...


def supports_strict_schema(model: str) -> bool:
    return model.lower().startswith(STRICT_SCHEMA_PREFIXES)


def _error_body(exc: openai.APIStatusError) -> dict[str, Any]:
    body = exc.body if isinstance(exc.body, dict) else {}
    return body.get("error", body) if isinstance(body.get("error"), dict) else body


def _retry_after(exc: openai.APIStatusError) -> float | None:
    try:
        value = exc.response.headers.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float,
        temperature: float,
        max_tokens: int = 1200,
        response_format: str = "auto",
        reasoning_effort: str | None = None,
        max_retry_wait: float = 2.0,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.max_retry_wait = max_retry_wait
        self._sleep = sleep
        # max_retries=0: retries are handled below so a long retry-after never blocks a request.
        self._client = OpenAI(
            api_key=api_key or "not-needed",  # keyless local providers still need a string
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
            http_client=http_client,
        )
        if response_format == "auto":
            first = "json_schema" if supports_strict_schema(model) else "json_object"
        else:
            first = response_format
        order = ["json_schema", "json_object", "none"]
        self._formats = order[order.index(first):]

    @property
    def response_format(self) -> str:
        return self._formats[0]

    # ----------------------------------------------------------------- public

    def complete(self, messages: Messages, *, schema: dict | None = None) -> LLMResult:
        for attempt in (1, 2):
            try:
                return self._create(messages, schema)
            except openai.RateLimitError as exc:
                wait = _retry_after(exc)
                if attempt == 1 and wait is not None and wait <= self.max_retry_wait:
                    log.info("LLM rate limited; retrying after %.1fs", wait)
                    self._sleep(wait)
                    continue
                hint = f" (retry after {wait:g}s)" if wait is not None else ""
                raise LLMRequestError(f"LLM provider rate limit reached{hint}", kind="rate_limit") from exc
            except openai.APITimeoutError as exc:
                raise LLMTimeoutError(f"LLM did not respond within {self.timeout:g}s") from exc
            except (openai.InternalServerError, openai.APIConnectionError) as exc:
                if attempt == 1:
                    self._sleep(0.5)
                    continue
                if isinstance(exc, openai.APIConnectionError):
                    raise LLMRequestError("Could not reach the LLM provider; check LLM_BASE_URL", kind="connection") from exc
                raise LLMRequestError(f"LLM provider returned HTTP {exc.status_code}", kind="http") from exc
        raise AssertionError("unreachable")  # pragma: no cover

    # ----------------------------------------------------------------- internals

    def _create(self, messages: Messages, schema: dict | None) -> LLMResult:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        mode = self._formats[0]
        if mode == "json_schema" and schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "restaurant_ranking", "strict": True, "schema": schema},
            }
        elif mode in ("json_schema", "json_object"):
            kwargs["response_format"] = {"type": "json_object"}
        if self.reasoning_effort and self.model.lower().startswith(REASONING_PREFIXES):
            kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}

        try:
            raw = self._client.chat.completions.with_raw_response.create(**kwargs)
        except openai.BadRequestError as exc:
            body = _error_body(exc)
            code = str(body.get("code") or exc.code or "")
            if code == "json_validate_failed":
                # Groq could not produce valid JSON: let the service run its repair turn.
                raise LLMParseError(
                    "the provider could not produce valid JSON", raw=body.get("failed_generation")
                ) from exc
            message = str(body.get("message") or exc).lower()
            if "response_format" in kwargs and len(self._formats) > 1 and (
                "response_format" in message or "json_schema" in message or "schema" in message
            ):
                dropped = self._formats.pop(0)
                log.info("Provider rejected %s output; retrying with %s", dropped, self._formats[0])
                return self._create(messages, schema)
            raise LLMRequestError(f"LLM provider rejected the request (HTTP {exc.status_code})", kind="bad_request") from exc
        except openai.AuthenticationError as exc:
            raise LLMRequestError("LLM authentication failed; check GROQ_API_KEY / LLM_API_KEY", kind="auth") from exc
        except openai.PermissionDeniedError as exc:
            raise LLMRequestError("LLM provider denied access to the model", kind="auth") from exc
        except openai.NotFoundError as exc:
            raise LLMRequestError(
                f"LLM model or endpoint not found ({self.model}); check LLM_MODEL / LLM_BASE_URL", kind="not_found"
            ) from exc
        except (openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError, openai.APIConnectionError):
            raise  # handled (and retried) in complete()
        except openai.APIStatusError as exc:
            raise LLMRequestError(f"LLM provider returned HTTP {exc.status_code}", kind="http") from exc
        except openai.OpenAIError as exc:
            raise LLMRequestError(f"LLM call failed ({type(exc).__name__})", kind="http") from exc

        response = raw.parse()
        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise LLMResponseError("LLM returned an empty response")
        usage = getattr(response, "usage", None)
        return LLMResult(
            text=content,
            model=getattr(response, "model", None) or self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            rate_limit={h: raw.headers[h] for h in _RATE_LIMIT_HEADERS if h in raw.headers},
        )


def build_client(settings: Settings, *, model: str | None = None, **overrides: Any) -> ChatClient | None:
    """Return a client, or None when no LLM is configured (fallback-only mode)."""
    if not settings.llm_enabled:
        return None
    options: dict[str, Any] = {
        "api_key": settings.llm_api_key.get_secret_value().strip(),
        "base_url": settings.resolved_base_url,
        "model": model or settings.resolved_model,
        "timeout": settings.llm_timeout_seconds,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "response_format": settings.llm_response_format,
        "reasoning_effort": settings.llm_reasoning_effort,
        "max_retry_wait": settings.llm_max_retry_wait_seconds,
    }
    options.update(overrides)
    return OpenAICompatibleClient(**options)


def as_result(output: LLMResult | str) -> LLMResult:
    """Normalize a client's return value (test doubles may return plain strings)."""
    return output if isinstance(output, LLMResult) else LLMResult(text=output)
