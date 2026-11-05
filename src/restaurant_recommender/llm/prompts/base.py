"""Shared pieces for prompt variants: the variant type, size bounding, repair turn, schema."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from restaurant_recommender.data.budget import band_range_label
from restaurant_recommender.models.schemas import PreferenceRequest, Restaurant

# Upper bound on the user prompt size; candidates are dropped from the bottom
# (lowest rated) if a shortlist would exceed it (edge case LM-17).
MAX_PROMPT_CHARS = 24_000


@dataclass(frozen=True)
class PromptVariant:
    name: str
    description: str
    system_prompt: str
    build_user_prompt: Callable[[PreferenceRequest, list[Restaurant], int], str]


def sanitize_notes(text: str) -> str:
    """Keep the user's notes inside their delimiter (edge case IN-12)."""
    return text.replace("<", "‹").replace(">", "›")


def budget_text(prefs: PreferenceRequest) -> str:
    return f"{prefs.budget.value} ({band_range_label(prefs.budget)})" if prefs.budget else "any"


def build_messages(
    variant: PromptVariant,
    prefs: PreferenceRequest,
    candidates: list[Restaurant],
    top_n: int,
    max_chars: int = MAX_PROMPT_CHARS,
) -> tuple[list[dict[str, str]], list[Restaurant]]:
    """Return chat messages and the candidates actually included in them."""
    included = list(candidates)
    user = variant.build_user_prompt(prefs, included, top_n)
    while len(user) > max_chars and len(included) > top_n:
        included.pop()
        user = variant.build_user_prompt(prefs, included, top_n)
    return [
        {"role": "system", "content": variant.system_prompt},
        {"role": "user", "content": user},
    ], included


def build_repair_messages(
    messages: list[dict[str, str]], bad_output: str, problem: str
) -> list[dict[str, str]]:
    """One follow-up turn asking the model to fix an unusable reply (edge case LM-06)."""
    return [
        *messages,
        {"role": "assistant", "content": (bad_output or "")[:4000] or "(no usable output)"},
        {
            "role": "user",
            "content": (
                f"Your previous reply could not be used: {problem}. Reply again with ONLY the JSON "
                'object {"summary": ..., "recommendations": [{"id": ..., "explanation": ...}]}, '
                "using only ids from the CANDIDATES list."
            ),
        },
    ]


def ranking_schema(candidate_ids: list[str]) -> dict:
    """JSON schema for strict structured outputs. Ids are an enum, so a model using
    constrained decoding cannot return a restaurant outside the shortlist (INV-01)."""
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": list(candidate_ids)},
                        "explanation": {"type": "string"},
                    },
                    "required": ["id", "explanation"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["summary", "recommendations"],
        "additionalProperties": False,
    }
