"""Explanation-quality judges (eval.md §C rubric: groundedness, preference fit, clarity; 0-2 each).

* `HeuristicJudge`: offline, deterministic approximation of the rubric.
* `LLMJudge`: Groq LLM-as-judge (default `openai/gpt-oss-120b`, a different model
  family from the ranker) with a strict JSON schema.

"Acceptable" = total >= 4 of 6.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from restaurant_recommender.data.budget import band_range_label
from restaurant_recommender.llm.client import ChatClient, as_result
from restaurant_recommender.llm.errors import LLMError
from restaurant_recommender.llm.parser import extract_json
from restaurant_recommender.models.schemas import PreferenceRequest, Restaurant

log = logging.getLogger(__name__)

ACCEPTABLE_TOTAL = 4
# Amenities the dataset has no field for: claiming them unprompted is ungrounded.
_UNSUPPORTED_AMENITIES = re.compile(
    r"\b(parking|rooftop|outdoor|terrace|wi-?fi|live music|kids?|child(?:ren)?|wheelchair|pet|valet|ambien\w*|romantic|quiet)\b",
    re.I,
)
_HEDGES = re.compile(r"not confirmed|doesn't confirm|does not confirm|no information|isn't listed|not listed|unverified|can't confirm|cannot confirm", re.I)
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


@dataclass
class JudgeScore:
    groundedness: int
    preference_fit: int
    clarity: int
    judge: str
    rationale: str = ""

    @property
    def total(self) -> int:
        return self.groundedness + self.preference_fit + self.clarity

    @property
    def acceptable(self) -> bool:
        return self.total >= ACCEPTABLE_TOTAL


def _mentions(text: str, phrase: str | None) -> bool:
    return bool(phrase) and re.search(rf"(?<!\w){re.escape(phrase.lower())}(?!\w)", text) is not None


def mentions_preference(text: str, prefs: PreferenceRequest, restaurant: Restaurant) -> bool:
    """True if the explanation refers to at least one hard preference dimension."""
    lowered = text.lower()
    if prefs.cuisine and any(_mentions(lowered, c) for c in [prefs.cuisine, *restaurant.cuisines]):
        return True
    if any(_mentions(lowered, p) for p in [restaurant.locality, *restaurant.areas, prefs.location.split(",")[0].strip()]):
        return True
    if prefs.budget and re.search(r"budget|afford|value|price|₹|cost|cheap|pricey|splurge", lowered):
        return True
    if prefs.min_rating and re.search(r"rat(ed|ing)|\b[0-5]\.\d\b|star", lowered):
        return True
    return False


class HeuristicJudge:
    name = "heuristic"

    def score(self, prefs: PreferenceRequest, restaurant: Restaurant, explanation: str) -> JudgeScore:
        text = explanation.strip()
        lowered = text.lower()

        facts = [
            restaurant.name, restaurant.locality, restaurant.rest_type, *restaurant.cuisines,
            f"{restaurant.rating:.1f}" if restaurant.rating is not None else None,
            f"{restaurant.cost_for_two:,}" if restaurant.cost_for_two else None,
            str(restaurant.cost_for_two) if restaurant.cost_for_two else None,
        ]
        dishes = [d.strip() for d in (restaurant.dish_liked or "").split(",") if d.strip()]
        cites_fact = any(_mentions(lowered, f) for f in [*facts, *dishes] if f)
        notes = prefs.additional_preferences.lower()
        risky = [m.group(0).lower() for m in _UNSUPPORTED_AMENITIES.finditer(text)]
        unsupported = [w for w in risky if w not in notes and not _HEDGES.search(text)]
        if unsupported:
            grounded = 0
        elif cites_fact:
            grounded = 2
        else:
            grounded = 1

        dims = 0
        if prefs.cuisine and any(_mentions(lowered, c) for c in [prefs.cuisine, *restaurant.cuisines]):
            dims += 1
        if any(_mentions(lowered, p) for p in [restaurant.locality, prefs.location.split(",")[0].strip()]):
            dims += 1
        if prefs.budget and re.search(r"budget|afford|value|price|₹|cost", lowered):
            dims += 1
        if prefs.min_rating and re.search(r"rat(ed|ing)|\b[0-5]\.\d\b", lowered):
            dims += 1
        if notes and any(w for w in re.findall(r"[a-z]{4,}", notes) if w in lowered):
            dims += 1
        fit = 2 if dims >= 2 else 1 if dims == 1 else 0

        sentences = max(1, len(_SENTENCE_END.findall(text)) or 1)
        words = len(text.split())
        clarity = 2 if sentences <= 3 and 6 <= words <= 70 else 1 if sentences <= 5 and words <= 120 else 0

        why = f"facts={'yes' if cites_fact else 'no'}, unsupported={unsupported or '-'}, dims={dims}, sentences={sentences}"
        return JudgeScore(grounded, fit, clarity, self.name, why)


JUDGE_SYSTEM = """\
You grade explanations written by a restaurant recommender. Be strict and literal.
Score each dimension 0, 1 or 2:
- groundedness: 0 = claims facts or amenities not in RESTAURANT FACTS; 1 = generic but safe; \
2 = specific and every claim is supported by the facts or the user's preferences. Saying that the \
data does not confirm something the user asked for (e.g. "quiet not confirmed by the data") is \
correct, honest behaviour and must NOT lower any score.
- preference_fit: 0 = ignores the user's preferences; 1 = loosely mentions one; \
2 = clearly connects the restaurant to the user's hard preferences (and notes, if any).
- clarity: 0 = incoherent; 1 = readable but long or vague; 2 = concise (1-3 sentences) and clear.
Return JSON only: {"groundedness": n, "preference_fit": n, "clarity": n, "rationale": "<one short sentence>"}"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "groundedness": {"type": "integer", "enum": [0, 1, 2]},
        "preference_fit": {"type": "integer", "enum": [0, 1, 2]},
        "clarity": {"type": "integer", "enum": [0, 1, 2]},
        "rationale": {"type": "string"},
    },
    "required": ["groundedness", "preference_fit", "clarity", "rationale"],
    "additionalProperties": False,
}


class LLMJudge:
    def __init__(self, client: ChatClient) -> None:
        self.client = client
        self.name = f"llm:{client.model}"
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.rate_limit: dict[str, str] = {}

    def score(self, prefs: PreferenceRequest, restaurant: Restaurant, explanation: str) -> JudgeScore | None:
        facts = {
            "name": restaurant.name,
            "cuisines": restaurant.cuisines,
            "rating": restaurant.rating,
            "votes": restaurant.votes,
            "cost_for_two_inr": restaurant.cost_for_two,
            "locality": restaurant.locality,
            "type": restaurant.rest_type,
            "table_booking": restaurant.book_table,
            "online_order": restaurant.online_order,
            "popular_dishes": restaurant.dish_liked,
        }
        user = "\n".join(
            [
                "USER PREFERENCES",
                f"location={prefs.location}; cuisine={prefs.cuisine or 'any'}; "
                f"budget={band_range_label(prefs.budget) if prefs.budget else 'any'}; "
                f"min_rating={prefs.min_rating or 'any'}; notes={prefs.additional_preferences or '(none)'}",
                "",
                "RESTAURANT FACTS",
                json.dumps({k: v for k, v in facts.items() if v is not None}, ensure_ascii=False),
                "",
                "EXPLANATION TO GRADE",
                explanation,
            ]
        )
        messages = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]
        try:
            result = as_result(self.client.complete(messages, schema=JUDGE_SCHEMA))
            data = extract_json(result.text)
        except (LLMError, ValueError) as exc:
            log.warning("judge failed: %s", exc)
            return None
        self.calls += 1
        self.prompt_tokens += result.prompt_tokens or 0
        self.completion_tokens += result.completion_tokens or 0
        self.rate_limit = result.rate_limit or self.rate_limit
        try:
            scores = [max(0, min(2, int(data[k]))) for k in ("groundedness", "preference_fit", "clarity")]
        except (KeyError, TypeError, ValueError):
            return None
        return JudgeScore(*scores, judge=self.name, rationale=str(data.get("rationale", ""))[:200])
