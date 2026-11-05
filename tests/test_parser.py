"""LLM output parsing (edge cases LM-05..LM-14, LM-18)."""

from __future__ import annotations

import json

import pytest

from restaurant_recommender.llm.errors import LLMParseError
from restaurant_recommender.llm.parser import MAX_EXPLANATION_CHARS, parse_ranking

ALLOWED = {"rest_a", "rest_b", "rest_c"}


def payload(*recs, summary="Nice picks.") -> str:
    return json.dumps({"summary": summary, "recommendations": list(recs)})


def test_valid_json():
    parsed = parse_ranking(payload({"id": "rest_b", "explanation": "Great."}, {"id": "rest_a", "explanation": "Good."}), ALLOWED, 5)
    assert parsed.items == [("rest_b", "Great."), ("rest_a", "Good.")]
    assert parsed.summary == "Nice picks."


def test_markdown_fences_are_stripped():  # LM-06
    raw = "```json\n" + payload({"id": "rest_a", "explanation": "x"}) + "\n```"
    assert parse_ranking(raw, ALLOWED, 5).items == [("rest_a", "x")]


def test_surrounding_prose_and_trailing_commas():  # LM-06 / LM-18
    raw = 'Sure! Here you go: {"summary": "s", "recommendations": [{"id": "rest_a", "explanation": "x",},],} Enjoy.'
    assert parse_ranking(raw, ALLOWED, 5).items == [("rest_a", "x")]


def test_hallucinated_ids_are_dropped():  # LM-08
    parsed = parse_ranking(payload({"id": "rest_zzz", "explanation": "fake"}, {"id": "rest_c", "explanation": "real"}), ALLOWED, 5)
    assert parsed.items == [("rest_c", "real")]
    assert parsed.dropped_ids == ["rest_zzz"]


def test_duplicates_keep_first():  # LM-12
    parsed = parse_ranking(payload({"id": "rest_a", "explanation": "1"}, {"id": "rest_a", "explanation": "2"}), ALLOWED, 5)
    assert parsed.items == [("rest_a", "1")] and parsed.duplicate_ids == ["rest_a"]


def test_more_than_top_n_is_truncated():  # LM-11
    recs = [{"id": i, "explanation": "x"} for i in ["rest_a", "rest_b", "rest_c"]]
    assert [i for i, _ in parse_ranking(payload(*recs), ALLOWED, 2).items] == ["rest_a", "rest_b"]


def test_missing_or_blank_explanation_is_none():  # LM-07 / LM-14
    parsed = parse_ranking(payload({"id": "rest_a"}, {"id": "rest_b", "explanation": "   "}), ALLOWED, 5)
    assert parsed.items == [("rest_a", None), ("rest_b", None)]


def test_explicit_ranks_are_honoured():  # LM-13
    raw = payload({"rank": 2, "id": "rest_a", "explanation": "x"}, {"rank": 1, "id": "rest_b", "explanation": "y"})
    assert [i for i, _ in parse_ranking(raw, ALLOWED, 5).items] == ["rest_b", "rest_a"]


def test_alternative_shapes():
    assert parse_ranking(json.dumps(["rest_a", "rest_b"]), ALLOWED, 5).items == [("rest_a", None), ("rest_b", None)]
    raw = json.dumps({"restaurants": [{"source_id": "rest_c", "reason": "r"}]})
    assert parse_ranking(raw, ALLOWED, 5).items == [("rest_c", "r")]


def test_long_explanations_are_trimmed():
    long_text = "This place is great. " * 60
    _, explanation = parse_ranking(payload({"id": "rest_a", "explanation": long_text}), ALLOWED, 5).items[0]
    assert len(explanation) <= MAX_EXPLANATION_CHARS + 1


@pytest.mark.parametrize(
    "raw",
    [
        "",  # LM-05
        "I recommend Toscano!",  # LM-18
        '{"summary": "x", "recommendations": [ {"id": ',  # truncated JSON
        json.dumps({"summary": "no list"}),  # wrong schema
        payload({"id": "rest_unknown", "explanation": "x"}),  # nothing valid
        "42",
    ],
)
def test_unusable_replies_raise(raw):
    with pytest.raises(LLMParseError):
        parse_ranking(raw, ALLOWED, 5)
