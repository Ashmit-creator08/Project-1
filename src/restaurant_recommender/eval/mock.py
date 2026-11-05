"""Offline LLM doubles for the eval harness (no network, deterministic)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from restaurant_recommender.llm.client import LLMResult

_ID = re.compile(r"\brest_[0-9a-f]{12}\b")


def prompt_ids(messages: list[dict[str, str]]) -> list[str]:
    user = next(m["content"] for m in messages if m["role"] == "user")
    return list(dict.fromkeys(_ID.findall(user)))


def _wanted(messages: list[dict[str, str]]) -> int:
    user = next(m["content"] for m in messages if m["role"] == "user")
    match = re.search(r"(?:best|exactly) (\d+)", user)
    return int(match.group(1)) if match else 5


class OfflineRanker:
    """Keeps the shortlist order (rating order) and writes a neutral explanation."""

    model = "offline-mock"

    def complete(self, messages, *, schema: dict | None = None) -> LLMResult:
        ids = prompt_ids(messages)[: _wanted(messages)]
        payload = {
            "summary": "Offline mock ranking (no LLM).",
            "recommendations": [{"id": rid, "explanation": "Offline mock explanation."} for rid in ids],
        }
        return LLMResult(text=json.dumps(payload), model=self.model)


class ScriptedLLM:
    """Replays responses (strings, exceptions or callables), then defers to `default`."""

    def __init__(self, *responses, default: Callable | None = None, model: str = "scripted-mock") -> None:
        self.responses = list(responses)
        self.default = default or OfflineRanker().complete
        self.model = model
        self.calls = 0

    def complete(self, messages, *, schema: dict | None = None):
        self.calls += 1
        response = self.responses.pop(0) if self.responses else self.default
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(messages, schema=schema) if response is self.default else response(messages)
        return response


def hallucinating(messages) -> str:
    """An invented id first, then the 2nd shortlisted id, then a duplicate."""
    ids = prompt_ids(messages)
    recs = [{"id": "rest_000000000000", "explanation": "Invented."}]
    if len(ids) > 1:
        recs += [{"id": ids[1], "explanation": "Real pick."}, {"id": ids[1], "explanation": "Duplicate."}]
    return json.dumps({"summary": "Picks.", "recommendations": recs})
