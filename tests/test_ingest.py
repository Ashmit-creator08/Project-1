"""Ingest parsing, cleaning, budget derivation and the Parquet store (edge cases DI-*)."""

from __future__ import annotations

import pandas as pd
import pytest

from restaurant_recommender.data.budget import LOW_MAX, MEDIUM_MAX, budget_band
from restaurant_recommender.data.ingest import (
    CANONICAL_COLUMNS,
    IngestError,
    SchemaError,
    clean,
    make_id,
    parse_bool,
    parse_cost,
    parse_rating,
    split_list,
    write_parquet_atomic,
)
from restaurant_recommender.data.store import DataUnavailableError, RestaurantStore
from restaurant_recommender.models.schemas import Budget


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("4.1/5", 4.1),
        ("3.9 /5", 3.9),
        (" 4 /5 ", 4.0),
        ("4.5", 4.5),
        (3.2, 3.2),
        ("NEW", None),
        ("-", None),
        ("", None),
        (None, None),
        (float("nan"), None),
        ("6/5", None),
        ("0/5", None),
    ],
)
def test_parse_rating(raw, expected):  # DI-06, DI-08
    assert parse_rating(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("800", 800),
        ("1,200", 1200),
        ("₹1,500", 1500),
        ("Rs. 300", 300),
        (450.0, 450),
        ("0", None),
        ("-100", None),
        ("abc", None),
        (None, None),
    ],
)
def test_parse_cost(raw, expected):  # DI-07, DI-09
    assert parse_cost(raw) == expected


def test_parse_bool():
    assert parse_bool("Yes") is True
    assert parse_bool(" no ") is False
    assert parse_bool("maybe") is None
    assert parse_bool(None) is None


def test_split_list_dedupes_and_trims():  # DI-10
    assert split_list("North Indian,  Mughlai , North Indian,,") == ["North Indian", "Mughlai"]
    assert split_list(None) == []


@pytest.mark.parametrize(
    ("cost", "band"),
    [
        (1, Budget.LOW),
        (LOW_MAX, Budget.LOW),  # FL-06: upper bound inclusive
        (LOW_MAX + 1, Budget.MEDIUM),
        (MEDIUM_MAX, Budget.MEDIUM),
        (MEDIUM_MAX + 1, Budget.HIGH),
        (100_000, Budget.HIGH),  # DI-11 outlier still banded
        (0, None),
        (None, None),
        (float("nan"), None),
    ],
)
def test_budget_band_boundaries(cost, band):
    assert budget_band(cost) == band


def test_make_id_is_deterministic_and_case_insensitive():  # DI-04
    assert make_id("Jalsa", "1 Road") == make_id(" jalsa ", "1 ROAD")
    assert make_id("Jalsa", "1 Road") != make_id("Jalsa", "2 Road")
    assert make_id("Jalsa", None).startswith("rest_")


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Jalsa", "Jalsa", "   ", "Cafe X", None],
            "address": ["1 Road, Banashankari, Bangalore"] * 2 + ["x", "2 Road, BTM, Bangalore", "y"],
            "location": ["Banashankari", "Banashankari", "BTM", "BTM", "BTM"],
            "listed_in(city)": ["Banashankari", "Jayanagar", "BTM", "BTM", "BTM"],
            "listed_in(type)": ["Buffet", "Delivery", "Delivery", "Cafes", "Delivery"],
            "cuisines": ["North Indian, Mughlai, Chinese", "North Indian, Mughlai, Chinese", "Cafe", "Cafe,  Italian ,Cafe", "x"],
            "rate": ["4.1/5", "4.0 /5", "3.0/5", "NEW", "-"],
            "votes": [775, 700, 1, 0, 0],
            "approx_cost(for two people)": ["800", "800", "300", "1,200", "x"],
            "rest_type": ["Casual Dining"] * 5,
            "online_order": ["Yes", "Yes", "No", "No", "No"],
            "book_table": ["Yes", "Yes", "No", "No", "No"],
            "dish_liked": ["Pasta", None, None, None, None],
            "reviews_list": ["[]"] * 5,  # unmapped columns are ignored
        }
    )


def test_clean_maps_dedupes_and_normalizes():
    df = clean(_raw_frame())
    assert list(df.columns) == CANONICAL_COLUMNS
    assert len(df) == 2  # duplicate listing merged, nameless rows dropped (DI-03, DI-05)

    jalsa = df[df["name"] == "Jalsa"].iloc[0]
    assert jalsa["rating"] == 4.1  # most-voted copy wins
    assert jalsa["votes"] == 775
    assert list(jalsa["areas"]) == ["Banashankari", "Jayanagar"]
    assert list(jalsa["listed_types"]) == ["Buffet", "Delivery"]
    assert list(jalsa["cuisines"]) == ["North Indian", "Mughlai", "Chinese"]
    assert jalsa["cost_for_two"] == 800 and jalsa["budget_band"] == "medium"
    assert jalsa["city"] == "Bangalore"
    assert bool(jalsa["book_table"]) is True

    cafe = df[df["name"] == "Cafe X"].iloc[0]
    assert pd.isna(cafe["rating"])  # "NEW" -> unrated, not 0
    assert list(cafe["cuisines"]) == ["Cafe", "Italian"]
    assert cafe["cost_for_two"] == 1200 and cafe["budget_band"] == "high"
    assert df["id"].is_unique


def test_clean_fails_fast_on_schema_drift():  # DI-02
    raw = _raw_frame().drop(columns=["rate", "cuisines"])
    with pytest.raises(SchemaError, match="cuisines.*rate|rate.*cuisines"):
        clean(raw)


def test_clean_rejects_dataset_without_usable_rows():  # DI-12
    raw = _raw_frame()
    raw["name"] = None
    with pytest.raises(IngestError):
        clean(raw)


def test_parquet_round_trip_and_atomic_write(tmp_path):  # DI-15
    out = tmp_path / "processed" / "restaurants.parquet"
    df = clean(_raw_frame())
    write_parquet_atomic(df, out)
    assert out.exists() and not (out.parent / "restaurants.parquet.tmp").exists()

    store = RestaurantStore.from_parquet(out)
    assert len(store) == 2
    jalsa = next(r for r in store.get_many(store.get_all()["id"]) if r.name == "Jalsa")
    assert jalsa.areas == ["Banashankari", "Jayanagar"]
    assert jalsa.budget_band is Budget.MEDIUM and jalsa.cost_for_two == 800
    assert store.get("rest_unknown") is None


def test_store_missing_file_is_a_clear_error(tmp_path):  # DI-13
    with pytest.raises(DataUnavailableError, match="ingest"):
        RestaurantStore.from_parquet(tmp_path / "nope.parquet")


def test_store_corrupt_file_is_a_clear_error(tmp_path):  # DI-14
    bad = tmp_path / "restaurants.parquet"
    bad.write_bytes(b"not a parquet file")
    with pytest.raises(DataUnavailableError, match="Could not read"):
        RestaurantStore.from_parquet(bad)


def test_store_rejects_empty_or_incomplete_frames(restaurants_df):
    with pytest.raises(DataUnavailableError, match="empty"):
        RestaurantStore(restaurants_df.iloc[0:0])
    with pytest.raises(DataUnavailableError, match="missing columns"):
        RestaurantStore(restaurants_df.drop(columns=["rating"]))


def test_store_options(store):
    assert store.cities() == ["Bangalore"]
    assert store.localities()[0] == "Indiranagar"  # most common first
    assert "Italian" in store.cuisines()
