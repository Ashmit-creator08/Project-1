"""Shared test fixtures data and a scriptable fake LLM client."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

import pandas as pd

from restaurant_recommender.data.budget import budget_band
from restaurant_recommender.data.ingest import make_id

# (name, locality, cuisines, rating, cost_for_two, votes)
ROWS = [
    ("Toscano", "Indiranagar", ["Italian"], 4.5, 1200, 900),
    ("Pasta Street", "Indiranagar", ["Italian", "Continental"], 4.1, 800, 500),
    ("Pizza Bakery", "Koramangala 5th Block", ["Pizza", "Italian"], 3.9, 600, 300),
    ("Punjabi Tadka", "Indiranagar", ["North Indian", "Chinese"], 4.0, 500, 700),
    ("Wok Express", "BTM", ["Chinese"], 3.6, 300, 200),
    ("Udupi Grand", "Jayanagar", ["South Indian"], 4.6, 200, 1500),
    ("New Cafe", "Koramangala 5th Block", ["Cafe", "Italian"], None, 700, 0),
    ("Steak House", "Indiranagar", ["Steak"], 4.2, None, 400),
    ("Thai Garden", "Koramangala 7th Block", ["Chinese", "Thai"], 3.5, 900, 250),
    ("Chai Point", "Jayanagar", ["Tea", "Beverages"], 3.8, 250, 600),
    ("Olive Bistro", "Whitefield", ["Italian"], 3.5, 1000, 150),
    ("Dhaba 24", "Whitefield", ["North Indian"], 2.9, 400, 80),
]


def make_row(name, locality, cuisines, rating, cost, votes=100, **extra) -> dict:
    address = extra.pop("address", f"12, Main Road, {locality}, Bangalore")
    band = budget_band(cost)
    row = {
        "id": make_id(name, address),
        "name": name,
        "city": "Bangalore",
        "locality": locality,
        "areas": [locality],
        "address": address,
        "cuisines": list(cuisines),
        "rating": rating,
        "votes": votes,
        "cost_for_two": cost,
        "budget_band": band.value if band else None,
        "rest_type": "Casual Dining",
        "online_order": True,
        "book_table": False,
        "dish_liked": "Pasta, Pizza" if "Italian" in cuisines else None,
        "listed_types": ["Delivery"],
    }
    row.update(extra)
    return row


def make_df(rows=ROWS) -> pd.DataFrame:
    df = pd.DataFrame([make_row(*r) for r in rows])
    df["cost_for_two"] = pd.array(df["cost_for_two"].tolist(), dtype="Int64")
    return df


# --------------------------------------------------------------------------- fake LLM

_ID_IN_PROMPT = re.compile(r"\brest_[0-9a-f]{12}\b")


def prompt_ids(messages: list[dict[str, str]]) -> list[str]:
    """Candidate ids in prompt order (best rated first); works for every prompt variant."""
    user = next(m["content"] for m in messages if m["role"] == "user")
    return list(dict.fromkeys(_ID_IN_PROMPT.findall(user)))


def reverse_ranker(messages: list[dict[str, str]]) -> str:
    """Ranks candidates worst-first and tries to inject fake facts (which must be ignored)."""
    ids = prompt_ids(messages)
    return json.dumps(
        {
            "summary": "A strong set of picks for your evening.",
            "recommendations": [
                {"id": rid, "explanation": f"Chosen for you ({rid}).", "name": "INVENTED NAME", "rating": 5.0}
                for rid in reversed(ids)
            ],
        }
    )


def hallucinating_ranker(messages: list[dict[str, str]]) -> str:
    """Puts an invented id first, then the 2nd candidate, then a duplicate."""
    ids = prompt_ids(messages)
    recs = [{"id": "rest_doesnotexist", "explanation": "Invented."}]
    if len(ids) > 1:
        recs += [{"id": ids[1], "explanation": "Real pick."}, {"id": ids[1], "explanation": "Dup."}]
    return json.dumps({"summary": "Picks.", "recommendations": recs})


class FakeLLM:
    """Returns scripted responses in order; then falls back to `default`.

    A response may be a string, an exception instance (raised) or a callable
    taking the messages.
    """

    model = "fake-llm"

    def __init__(self, *responses, default: Callable | str = reverse_ranker) -> None:
        self.responses = list(responses)
        self.default = default
        self.calls: list[list[dict[str, str]]] = []
        self.schemas: list[dict | None] = []

    def complete(self, messages, *, schema: dict | None = None):
        self.calls.append(messages)
        self.schemas.append(schema)
        response = self.responses.pop(0) if self.responses else self.default
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(messages)
        return response
