"""Application settings loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root: Apps/ (parent of src/)
REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_PATH = REPO_ROOT / "data" / "processed" / "restaurants.parquet"

# Hard upper bound for `top_n` in requests (edge case IN-06: reject above max).
MAX_TOP_N = 20

# Hosts that serve an OpenAI-compatible API without needing a key (e.g. Ollama, LM Studio).
_KEYLESS_HOSTS = {"localhost", "127.0.0.1", "::1"}

# Provider presets (Phase 4: Groq is the default). Any field can be overridden by env.
# Groq: gpt-oss-120b ranks (production, strict json_schema); gpt-oss-20b judges in evals.
# llama-3.3-70b-versatile is listed by Groq but not available to every account (404),
# so it is not the default; set LLM_MODEL to use it where it is enabled.
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-120b",
        "judge_model": "openai/gpt-oss-20b",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "judge_model": "gpt-4o-mini",
    },
}

Provider = Literal["groq", "openai", "custom"]
ResponseFormat = Literal["auto", "json_schema", "json_object", "none"]


class Settings(BaseSettings):
    """Runtime configuration for the restaurant recommender."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider ------------------------------------------------------------
    llm_provider: Provider = Field(default="groq", alias="LLM_PROVIDER")
    # SecretStr keeps the key out of reprs and logs (edge case CF-04).
    llm_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias=AliasChoices("LLM_API_KEY", "GROQ_API_KEY")
    )
    llm_base_url: str | None = Field(default=None, alias="LLM_BASE_URL")
    llm_model: str | None = Field(default=None, alias="LLM_MODEL")
    judge_model: str | None = Field(default=None, alias="JUDGE_MODEL")
    llm_timeout_seconds: float = Field(default=20.0, alias="LLM_TIMEOUT_SECONDS", gt=0, le=120)
    llm_temperature: float = Field(default=0.3, alias="LLM_TEMPERATURE", ge=0, le=1)
    llm_max_tokens: int = Field(default=1200, alias="LLM_MAX_TOKENS", ge=100, le=8000)
    llm_response_format: ResponseFormat = Field(default="auto", alias="LLM_RESPONSE_FORMAT")
    # Used only for reasoning models (gpt-oss); keeps hidden reasoning tokens (and TPM use) low.
    llm_reasoning_effort: Literal["low", "medium", "high"] | None = Field(
        default="low", alias="LLM_REASONING_EFFORT"
    )
    # Honour a provider's 429 retry-after only when it is this short; otherwise fall back.
    llm_max_retry_wait_seconds: float = Field(default=2.0, alias="LLM_MAX_RETRY_WAIT_SECONDS", ge=0, le=30)
    # v2_stricter_json won the live Groq A/B (2026-11-05): mean judge score 5.60 vs 5.07,
    # 47% fewer prompt tokens, all hard metrics 100% for both. See Docs/implementation-plan.md.
    prompt_variant: str = Field(default="v2_stricter_json", alias="PROMPT_VARIANT")

    # --- Data ----------------------------------------------------------------------
    data_path: Path = Field(default=_DEFAULT_DATA_PATH, alias="DATA_PATH")
    max_candidates: int = Field(default=30, alias="MAX_CANDIDATES", ge=1, le=100)
    top_n_default: int = Field(default=5, alias="TOP_N_DEFAULT", ge=1, le=MAX_TOP_N)

    # --- Phase 4 hardening -------------------------------------------------------------
    cache_ttl_seconds: int = Field(default=600, alias="CACHE_TTL_SECONDS", ge=0)
    cache_max_entries: int = Field(default=256, alias="CACHE_MAX_ENTRIES", ge=1)
    rate_limit_per_minute: int = Field(default=30, alias="RATE_LIMIT_PER_MINUTE", ge=0)

    # --- Deployment --------------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO", alias="LOG_LEVEL")
    log_format: Literal["text", "json"] = Field(default="text", alias="LOG_FORMAT")
    # Comma-separated browser origins allowed to call the API directly; empty = CORS off.
    cors_allow_origins: str = Field(default="", alias="CORS_ALLOW_ORIGINS")
    # When set, GET /metrics requires "Authorization: Bearer <token>".
    metrics_token: SecretStr = Field(default=SecretStr(""), alias="METRICS_TOKEN")

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @field_validator("data_path")
    @classmethod
    def _resolve_data_path(cls, value: Path) -> Path:
        # Relative paths are relative to the repo root, not the current directory,
        # so the API and UI find the same file wherever they are launched from.
        return value if value.is_absolute() else REPO_ROOT / value

    @field_validator("llm_base_url", "llm_model", "judge_model", mode="before")
    @classmethod
    def _blank_is_unset(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    # --- resolved provider values -----------------------------------------------------

    def _preset(self, key: str) -> str | None:
        return PROVIDER_PRESETS.get(self.llm_provider, {}).get(key)

    @property
    def resolved_base_url(self) -> str:
        return self.llm_base_url or self._preset("base_url") or PROVIDER_PRESETS["openai"]["base_url"]

    @property
    def resolved_model(self) -> str:
        return self.llm_model or self._preset("model") or PROVIDER_PRESETS["openai"]["model"]

    @property
    def resolved_judge_model(self) -> str:
        return self.judge_model or self._preset("judge_model") or self.resolved_model

    @property
    def llm_enabled(self) -> bool:
        """True when an LLM call can be attempted.

        Without a key (and without a local keyless provider) the app runs in
        fallback-only mode: rating-ranked results with template explanations (CF-01).
        """
        if self.llm_api_key.get_secret_value().strip():
            return True
        host = (urlparse(self.resolved_base_url).hostname or "").lower()
        return host in _KEYLESS_HOSTS


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
