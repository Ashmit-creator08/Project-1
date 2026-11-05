"""Download, clean and cache the Hugging Face Zomato dataset.

Usage::

    # Default: read only the needed columns from Hugging Face's Parquet export
    python -m restaurant_recommender.data.ingest

    # Offline / reproducible: ingest a local copy (CSV or Parquet with the raw columns)
    python -m restaurant_recommender.data.ingest --source path/to/zomato.csv

    # Full download through the `datasets` library (~575 MB CSV)
    python -m restaurant_recommender.data.ingest --method datasets

Output: ``data/processed/restaurants.parquet`` (or ``DATA_PATH``), one row per
unique restaurant in the canonical schema (see ``CANONICAL_COLUMNS``).

The raw dataset lists the same restaurant once per Zomato listing (e.g. under
"Delivery" and "Dine-out" in several zones), so rows are deduplicated by
name + address. The row with the most votes wins for scalar fields; the
listing zones and listing types of all copies are merged.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from restaurant_recommender.config import get_settings
from restaurant_recommender.data.budget import budget_band

log = logging.getLogger(__name__)

DATASET_ID = "ManikaSaini/zomato-restaurant-recommendation"
# Hugging Face's auto-converted Parquet export of the dataset (revision refs/convert/parquet).
HF_PARQUET_DIR = f"datasets/{DATASET_ID}@refs%2Fconvert%2Fparquet/default/train"

# Single mapping table from raw dataset columns to canonical roles (plan §1.1).
RAW_COLUMNS: dict[str, str] = {
    "name": "name",
    "address": "address",
    "location": "locality",
    "listed_in(city)": "area",
    "listed_in(type)": "listed_type",
    "cuisines": "cuisines",
    "rate": "rating",
    "votes": "votes",
    "approx_cost(for two people)": "cost_for_two",
    "rest_type": "rest_type",
    "online_order": "online_order",
    "book_table": "book_table",
    "dish_liked": "dish_liked",
}
REQUIRED_RAW_COLUMNS = {
    "name",
    "address",
    "location",
    "cuisines",
    "rate",
    "approx_cost(for two people)",
}
# The dataset only covers Bangalore; every row gets this city.
CITY = "Bangalore"

CANONICAL_COLUMNS = [
    "id",
    "name",
    "city",
    "locality",
    "areas",
    "address",
    "cuisines",
    "rating",
    "votes",
    "cost_for_two",
    "budget_band",
    "rest_type",
    "online_order",
    "book_table",
    "dish_liked",
    "listed_types",
]


class IngestError(RuntimeError):
    """Ingest could not produce a valid dataset. Nothing is written when this is raised."""


class SchemaError(IngestError):
    """The source dataset does not have the expected columns (edge case DI-02)."""


# --------------------------------------------------------------------------- parsing helpers

_WS = re.compile(r"\s+")
_RATING = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:/\s*5(?:\.0)?)?\s*$")
_COST_JUNK = re.compile(r"[₹,\s]|rs\.?|inr", re.IGNORECASE)


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    return isinstance(value, float) and math.isnan(value)


def clean_text(value: Any) -> str | None:
    """Strip and collapse whitespace; empty -> None."""
    if _is_missing(value):
        return None
    text = _WS.sub(" ", str(value)).strip()
    return text or None


def parse_rating(value: Any) -> float | None:
    """'4.1/5' -> 4.1, '3.9 /5' -> 3.9, 'NEW' / '-' / None -> None (edge cases DI-06, DI-08)."""
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        match = _RATING.match(str(value))
        if not match:
            return None
        number = float(match.group(1))
    return round(number, 1) if 0 < number <= 5 else None


def parse_cost(value: Any) -> int | None:
    """'1,200' / '₹800' / 800.0 -> int. Unparseable or <= 0 -> None (edge cases DI-07, DI-09).

    A cost of 0 is treated as missing data rather than "free", so it gets no
    budget band and displays as N/A.
    """
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = _COST_JUNK.sub("", str(value))
        try:
            number = float(text)
        except ValueError:
            return None
    if math.isnan(number) or number <= 0:
        return None
    return int(round(number))


def parse_bool(value: Any) -> bool | None:
    text = clean_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered in {"yes", "y", "true", "1"}:
        return True
    if lowered in {"no", "n", "false", "0"}:
        return False
    return None


def split_list(value: Any) -> list[str]:
    """'North Indian, Mughlai,  Chinese' -> ['North Indian', 'Mughlai', 'Chinese'] (edge case DI-10)."""
    text = clean_text(value)
    if text is None:
        return []
    items = (clean_text(part) for part in text.split(","))
    return list(dict.fromkeys(item for item in items if item))


def make_id(name: str, address: str | None) -> str:
    """Deterministic id from name + address (edge case DI-04: source has no id)."""
    key = f"{name.strip().lower()}|{(address or '').strip().lower()}"
    return "rest_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- loading


def load_raw_from_hf(method: str = "parquet") -> pd.DataFrame:
    """Load the raw dataset from Hugging Face.

    ``parquet`` reads only the mapped columns from the Parquet export (range
    requests, skips the large review/menu text). ``datasets`` downloads the full
    CSV through the `datasets` library.
    """
    columns = list(RAW_COLUMNS)
    try:
        if method == "datasets":
            from datasets import load_dataset

            dataset = load_dataset(DATASET_ID, split="train")
            keep = [c for c in columns if c in dataset.column_names]
            return dataset.select_columns(keep).to_pandas()

        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        files = sorted(fs.glob(f"{HF_PARQUET_DIR}/*.parquet"))
        if not files:
            raise IngestError(f"No Parquet files found for {DATASET_ID} on Hugging Face.")
        frames = []
        for path in files:
            log.info("Reading %s", path)
            with fs.open(path, "rb") as handle:
                available = set(pq.ParquetFile(handle).schema_arrow.names)
                handle.seek(0)
                keep = [c for c in columns if c in available]
                frames.append(pq.read_table(handle, columns=keep).to_pandas())
        return pd.concat(frames, ignore_index=True)
    except IngestError:
        raise
    except Exception as exc:  # network, auth, missing package... (edge case DI-01)
        raise IngestError(
            f"Could not load {DATASET_ID} from Hugging Face ({type(exc).__name__}: {exc}). "
            "Check your connection, or pass --source with a local copy."
        ) from exc


def load_raw_from_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise IngestError(f"Source file not found: {path}")
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path, usecols=lambda c: c in RAW_COLUMNS, low_memory=False)
    except Exception as exc:
        raise IngestError(f"Could not read {path}: {exc}") from exc


# --------------------------------------------------------------------------- cleaning


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw columns to the canonical schema, parse types and deduplicate."""
    missing = REQUIRED_RAW_COLUMNS - set(raw.columns)
    if missing:
        raise SchemaError(
            f"Source dataset is missing expected columns: {sorted(missing)}. "
            "Update RAW_COLUMNS in ingest.py to match the new schema."
        )

    df = pd.DataFrame(index=raw.index)
    for raw_col, role in RAW_COLUMNS.items():
        df[role] = raw[raw_col] if raw_col in raw.columns else None
    df = df.astype(object)

    df["name"] = df["name"].map(clean_text)
    df = df[df["name"].notna()]  # edge case DI-05: never keep nameless rows
    if df.empty:
        raise IngestError("No usable rows after cleaning (all names empty).")

    df["address"] = df["address"].map(clean_text)
    df["locality"] = df["locality"].map(clean_text)
    df["area"] = df["area"].map(clean_text)
    df["listed_type"] = df["listed_type"].map(clean_text)
    df["rest_type"] = df["rest_type"].map(clean_text)
    df["dish_liked"] = df["dish_liked"].map(clean_text)
    df["cuisines"] = df["cuisines"].map(split_list)
    df["rating"] = df["rating"].map(parse_rating)
    df["cost_for_two"] = df["cost_for_two"].map(parse_cost)
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype("int64")
    df["online_order"] = df["online_order"].map(parse_bool)
    df["book_table"] = df["book_table"].map(parse_bool)
    df["id"] = [make_id(n, a) for n, a in zip(df["name"], df["address"])]

    # Dedupe (edge case DI-03): most-voted copy wins; merge zones/types of all copies.
    areas: dict[str, set[str]] = defaultdict(set)
    listed_types: dict[str, set[str]] = defaultdict(set)
    for rid, area, listed_type in zip(df["id"], df["area"], df["listed_type"]):
        if area:
            areas[rid].add(area)
        if listed_type:
            listed_types[rid].add(listed_type)
    df = df.sort_values(["votes", "rating"], ascending=[False, False], na_position="last", kind="stable")
    df = df.drop_duplicates("id", keep="first").reset_index(drop=True)
    df["areas"] = pd.Series([sorted(areas[r]) for r in df["id"]], index=df.index, dtype=object)
    df["listed_types"] = pd.Series(
        [sorted(listed_types[r]) for r in df["id"]], index=df.index, dtype=object
    )

    df["city"] = CITY
    df["budget_band"] = df["cost_for_two"].map(_band_value)

    out = df[CANONICAL_COLUMNS].copy()
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce").astype("float64")
    out["cost_for_two"] = pd.array(out["cost_for_two"].tolist(), dtype="Int64")
    out["online_order"] = pd.array(out["online_order"].tolist(), dtype="boolean")
    out["book_table"] = pd.array(out["book_table"].tolist(), dtype="boolean")
    out = out.sort_values(["rating", "votes", "name"], ascending=[False, False, True], na_position="last")
    out = out.reset_index(drop=True)

    if out.empty:  # edge case DI-12
        raise IngestError("Dataset is empty after cleaning.")
    if not out["id"].is_unique:
        raise IngestError("Generated restaurant ids are not unique.")
    return out


def _band_value(cost: Any) -> str | None:
    band = budget_band(cost)
    return band.value if band else None


def summarize(df: pd.DataFrame, raw_rows: int) -> dict[str, Any]:
    bands = Counter(df["budget_band"].fillna("none"))
    return {
        "raw_rows": raw_rows,
        "restaurants": len(df),
        "unrated": int(df["rating"].isna().sum()),
        "missing_cost": int(df["cost_for_two"].isna().sum()),
        "budget_bands": dict(sorted(bands.items())),
        "localities": int(df["locality"].nunique()),
        "median_cost_for_two": float(df["cost_for_two"].median()),
    }


# --------------------------------------------------------------------------- writing


def write_parquet_atomic(df: pd.DataFrame, path: Path, retries: int = 3) -> None:
    """Write to a temp file, then rename over the target (edge cases DI-15, DI-16)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    for attempt in range(1, retries + 1):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Windows / OneDrive can briefly lock a file that a reader has open.
            if attempt == retries:
                tmp.unlink(missing_ok=True)
                raise IngestError(
                    f"Could not replace {path}: file is locked (close the API/UI or pause sync)."
                ) from None
            time.sleep(0.5 * attempt)


def ingest(
    source: Path | None = None,
    method: str = "parquet",
    out_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out_path = out_path or get_settings().data_path
    raw = load_raw_from_file(source) if source else load_raw_from_hf(method)
    df = clean(raw)
    write_parquet_atomic(df, out_path)
    return df, summarize(df, len(raw))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, help="Local CSV/Parquet with the raw dataset columns")
    parser.add_argument(
        "--method",
        choices=["parquet", "datasets"],
        default="parquet",
        help="How to fetch from Hugging Face when --source is not given (default: parquet)",
    )
    parser.add_argument("--out", type=Path, help="Output Parquet path (default: DATA_PATH)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    print(f"Fetching {DATASET_ID} (this can take a few minutes)…" if not args.source else f"Reading {args.source}…")
    started = time.perf_counter()
    try:
        _, stats = ingest(args.source, args.method, args.out)
    except IngestError as exc:
        print(f"Ingest failed: {exc}", file=sys.stderr)
        return 1
    out = args.out or get_settings().data_path
    print(f"Wrote {stats['restaurants']:,} restaurants to {out} in {time.perf_counter() - started:.1f}s")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
