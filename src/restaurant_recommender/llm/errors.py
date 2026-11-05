"""Error types raised by the LLM layer. Any `LLMError` triggers the rating fallback."""

from __future__ import annotations


class LLMError(Exception):
    """Base class: the LLM could not produce a usable ranking.

    Messages are safe to show and log; they never contain the API key.
    `kind` is a short machine-readable category used by telemetry.
    """

    kind = "error"

    def __init__(self, message: str, *, kind: str | None = None) -> None:
        super().__init__(message)
        if kind:
            self.kind = kind


class LLMTimeoutError(LLMError):
    """The provider did not answer within LLM_TIMEOUT_SECONDS (edge case LM-01)."""

    kind = "timeout"


class LLMRequestError(LLMError):
    """Auth, rate-limit, 5xx or connection failure (edge cases LM-02..LM-04).

    `kind` is one of: auth, rate_limit, http, connection, bad_request, not_found.
    """

    kind = "http"


class LLMResponseError(LLMError):
    """The provider answered, but with nothing usable (edge case LM-05)."""

    kind = "empty"


class LLMParseError(LLMResponseError):
    """The answer was not valid JSON in the expected shape (edge cases LM-06, LM-07, LM-18).

    `raw` holds the unusable output when available (e.g. Groq's `failed_generation`),
    so the repair turn can show it back to the model.
    """

    kind = "parse"

    def __init__(self, message: str, *, raw: str | None = None) -> None:
        super().__init__(message)
        self.raw = raw
