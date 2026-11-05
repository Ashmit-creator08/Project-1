"""Parse and validate the LLM's ranking JSON against the candidate allowlist."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from restaurant_recommender.llm.errors import LLMParseError

MAX_EXPLANATION_CHARS = 500
MAX_SUMMARY_CHARS = 600

_FENCE = re.compile(r"^```[a-zA-Z]*\s*(.*?)\s*```$", re.DOTALL)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_WS = re.compile(r"\s+")


@dataclass
class ParsedRanking:
    items: list[tuple[str, str | None]]  # (candidate id, explanation or None) best first
    summary: str | None
    dropped_ids: list[str] = field(default_factory=list)  # not in the candidate set (LM-08)
    duplicate_ids: list[str] = field(default_factory=list)  # repeated (LM-12)


def extract_json(raw: str) -> Any:
    """Parse JSON, tolerating markdown fences, surrounding prose and trailing commas."""
    text = (raw or "").strip()
    if not text:
        raise LLMParseError("the reply was empty")
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise LLMParseError("the reply was not JSON")
    candidate = _TRAILING_COMMA.sub(r"\1", text[start : end + 1])
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMParseError(f"the reply was not valid JSON ({exc.msg})") from None


def clean_prose(value: Any, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = _WS.sub(" ", value).strip()
    if not text:
        return None
    if len(text) > limit:
        cut = text[:limit]
        text = (cut.rsplit(". ", 1)[0] + ".") if ". " in cut else cut.rsplit(" ", 1)[0] + "…"
    return text


def _entry_rank(entry: Any) -> float | None:
    if isinstance(entry, dict):
        rank = entry.get("rank")
        if isinstance(rank, (int, float)) and not isinstance(rank, bool):
            return float(rank)
    return None


def parse_ranking(raw: str, allowed_ids: set[str], top_n: int) -> ParsedRanking:
    data = extract_json(raw)

    summary: Any = None
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = next(
            (data[k] for k in ("recommendations", "restaurants", "results") if isinstance(data.get(k), list)),
            None,
        )
        summary = data.get("summary")
    else:
        entries = None
    if entries is None:
        raise LLMParseError("the reply had no 'recommendations' list")

    # Honour explicit ranks when every entry has one; otherwise list order (LM-13).
    ranks = [_entry_rank(e) for e in entries]
    if entries and all(r is not None for r in ranks):
        entries = [e for _, e in sorted(zip(ranks, entries), key=lambda pair: pair[0])]

    result = ParsedRanking(items=[], summary=clean_prose(summary, MAX_SUMMARY_CHARS))
    seen: set[str] = set()
    for entry in entries:
        if isinstance(entry, str):
            rid, explanation = entry, None
        elif isinstance(entry, dict):
            rid = entry.get("id") or entry.get("source_id") or entry.get("restaurant_id")
            explanation = entry.get("explanation") or entry.get("reason")
        else:
            continue
        if rid is None:
            continue
        rid = str(rid).strip()
        if rid not in allowed_ids:
            result.dropped_ids.append(rid)
            continue
        if rid in seen:
            result.duplicate_ids.append(rid)
            continue
        seen.add(rid)
        result.items.append((rid, clean_prose(explanation, MAX_EXPLANATION_CHARS)))
        if len(result.items) >= top_n:  # LM-11
            break

    if not result.items:
        raise LLMParseError("none of the recommended ids were in the candidate list")
    return result
