"""Prompt variants for ranking + explaining a filtered shortlist (Phase 4 prompt A/B).

Select one with ``PROMPT_VARIANT`` (default ``v1_baseline``); compare them with
``python -m restaurant_recommender.eval.run --prompt all``.
"""

from __future__ import annotations

from restaurant_recommender.llm.prompts import v1_baseline, v2_stricter_json
from restaurant_recommender.llm.prompts.base import (
    MAX_PROMPT_CHARS,
    PromptVariant,
    build_repair_messages,
    ranking_schema,
)
from restaurant_recommender.llm.prompts.base import build_messages as _build_messages
from restaurant_recommender.models.schemas import PreferenceRequest, Restaurant

VARIANTS: dict[str, PromptVariant] = {v.name: v for v in (v1_baseline.VARIANT, v2_stricter_json.VARIANT)}
DEFAULT_VARIANT = "v1_baseline"

# Backwards-compatible names (Phase 2 API).
PROMPT_VERSION = DEFAULT_VARIANT
SYSTEM_PROMPT = v1_baseline.SYSTEM_PROMPT
candidate_record = v1_baseline.candidate_record
build_user_prompt = v1_baseline.build_user_prompt


def get_variant(name: str | None) -> PromptVariant:
    try:
        return VARIANTS[name or DEFAULT_VARIANT]
    except KeyError:
        raise ValueError(f"Unknown PROMPT_VARIANT {name!r}; choose one of {sorted(VARIANTS)}") from None


def build_messages(
    prefs: PreferenceRequest,
    candidates: list[Restaurant],
    top_n: int,
    max_chars: int = MAX_PROMPT_CHARS,
    variant: str | PromptVariant | None = None,
) -> tuple[list[dict[str, str]], list[Restaurant]]:
    chosen = variant if isinstance(variant, PromptVariant) else get_variant(variant)
    return _build_messages(chosen, prefs, candidates, top_n, max_chars)


__all__ = [
    "DEFAULT_VARIANT",
    "MAX_PROMPT_CHARS",
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "VARIANTS",
    "PromptVariant",
    "build_messages",
    "build_repair_messages",
    "build_user_prompt",
    "candidate_record",
    "get_variant",
    "ranking_schema",
]
