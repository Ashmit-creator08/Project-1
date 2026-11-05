"""In-memory restaurant store backed by the cleaned Parquet snapshot."""

from __future__ import annotations

import hashlib
import logging
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from restaurant_recommender.data.ingest import CANONICAL_COLUMNS
from restaurant_recommender.data.locations import LocationIndex
from restaurant_recommender.models.schemas import Restaurant

log = logging.getLogger(__name__)

INGEST_HINT = "Run `python -m restaurant_recommender.data.ingest` to build it."
_LIST_COLUMNS = ("areas", "cuisines", "listed_types")


class DataUnavailableError(RuntimeError):
    """The restaurant snapshot is missing, unreadable or empty (edge cases DI-12..DI-14)."""


class RestaurantStore:
    """Read-only access to the canonical restaurant table.

    The DataFrame is never mutated after construction, so one store can serve
    concurrent requests (edge case OR-04).
    """

    def __init__(self, df: pd.DataFrame, source: str = "memory", version: str | None = None) -> None:
        missing = set(CANONICAL_COLUMNS) - set(df.columns)
        if missing:
            raise DataUnavailableError(
                f"Restaurant data at {source} is missing columns {sorted(missing)}. {INGEST_HINT}"
            )
        if df.empty:
            raise DataUnavailableError(f"Restaurant data at {source} is empty. {INGEST_HINT}")
        if not df["id"].is_unique:
            raise DataUnavailableError(f"Restaurant ids in {source} are not unique. {INGEST_HINT}")

        df = df[CANONICAL_COLUMNS].copy().reset_index(drop=True)
        for column in _LIST_COLUMNS:
            df[column] = pd.Series([_as_list(v) for v in df[column]], index=df.index, dtype=object)
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("float64")
        df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype("int64")

        # Location index over locality, Zomato listing zones and city. The street
        # address is deliberately excluded: it mentions landmarks such as
        # "Delhi Public School" that would make "Delhi" match Bangalore rows.
        self._locations = LocationIndex(
            [
                [p for p in (loc, *areas, city) if isinstance(p, str) and p]
                for loc, areas, city in zip(df["locality"], df["areas"], df["city"])
            ]
        )
        df["_cuisine_search"] = [", ".join(c).lower() for c in df["cuisines"]]

        self._df = df
        self._pos = {rid: i for i, rid in enumerate(df["id"])}
        counts: Counter[str] = Counter(c for row in df["cuisines"] for c in row)
        self._cuisines = [name for name, _ in counts.most_common()]
        self.source = source
        # Changes whenever the data does; part of the response-cache key (CF-06).
        self.version = version or hashlib.sha1("|".join(df["id"]).encode("utf-8")).hexdigest()[:12]

    # ----------------------------------------------------------------- loading

    @classmethod
    def from_parquet(cls, path: Path) -> RestaurantStore:
        path = Path(path)
        if not path.exists():
            raise DataUnavailableError(f"Restaurant data not found at {path}. {INGEST_HINT}")
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # corrupt file, lock, partial write...
            raise DataUnavailableError(
                f"Could not read restaurant data at {path} ({type(exc).__name__}: {exc}). {INGEST_HINT}"
            ) from exc
        stat = path.stat()
        store = cls(df, source=str(path), version=f"{stat.st_mtime_ns:x}-{stat.st_size:x}")
        log.info("Loaded %d restaurants from %s", len(store), path)
        return store

    # ----------------------------------------------------------------- access

    def __len__(self) -> int:
        return len(self._df)

    @property
    def frame(self) -> pd.DataFrame:
        """The full table, including private `_*_search` columns used by filters."""
        return self._df

    @property
    def locations(self) -> LocationIndex:
        return self._locations

    def get_all(self) -> pd.DataFrame:
        return self._df[CANONICAL_COLUMNS]

    def get(self, restaurant_id: str) -> Restaurant | None:
        pos = self._pos.get(restaurant_id)
        return None if pos is None else self._to_model(self._df.iloc[pos])

    def get_many(self, ids: Iterable[str]) -> list[Restaurant]:
        return [r for r in (self.get(i) for i in ids) if r is not None]

    def to_restaurants(self, frame: pd.DataFrame) -> list[Restaurant]:
        return [self._to_model(row) for _, row in frame.iterrows()]

    def localities(self) -> list[str]:
        """Known localities and listing zones, most common first (for UI suggestions)."""
        counts: Counter[str] = Counter()
        for locality, areas in zip(self._df["locality"], self._df["areas"]):
            names = {locality, *areas} if isinstance(locality, str) else set(areas)
            counts.update(n for n in names if n)
        return [name for name, _ in counts.most_common()]

    def cuisines(self) -> list[str]:
        """Known cuisines, most common first."""
        return list(self._cuisines)

    def cities(self) -> list[str]:
        return sorted(self._df["city"].dropna().unique().tolist())

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _to_model(row: pd.Series) -> Restaurant:
        data = {col: _clean_value(row[col]) for col in CANONICAL_COLUMNS}
        for column in _LIST_COLUMNS:
            data[column] = _as_list(row[column])
        data["votes"] = int(data["votes"] or 0)
        if data["cost_for_two"] is not None:
            data["cost_for_two"] = int(data["cost_for_two"])
        return Restaurant(**data)


def _clean_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _as_list(value: Any) -> list[str]:
    if value is None or value is pd.NA:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    return [str(v) for v in list(value) if v is not None and str(v)]
