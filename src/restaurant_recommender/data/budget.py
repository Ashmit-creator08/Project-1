"""Budget bands derived from the approximate cost for two people (INR).

Choice: **fixed thresholds** rather than dataset percentiles, so a band means the
same rupee range across re-ingests and can be shown to users as-is. They are
calibrated on the Bangalore Zomato data (12,494 unique restaurants, median cost
for two ₹400): about 59% low, 34% medium, 6% high, 0.5% unknown cost. Re-run
`ingest` to see the current split.

Boundaries are inclusive on the upper end (edge case FL-06):

    low     cost <= 400
    medium  400 < cost <= 1000
    high    cost > 1000

A missing or non-positive cost has no band, so such restaurants never match a
budget filter (edge case DI-07).
"""

from __future__ import annotations

import math

from restaurant_recommender.models.schemas import Budget

LOW_MAX = 400
MEDIUM_MAX = 1000


def budget_band(cost: float | int | None) -> Budget | None:
    if cost is None:
        return None
    try:
        value = float(cost)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or value <= 0:
        return None
    if value <= LOW_MAX:
        return Budget.LOW
    if value <= MEDIUM_MAX:
        return Budget.MEDIUM
    return Budget.HIGH


def band_range_label(band: Budget | str | None) -> str:
    """Human-readable rupee range, e.g. '₹401–₹1,000 for two'."""
    if band is None:
        return "any budget"
    band = Budget(band)
    if band is Budget.LOW:
        return f"up to ₹{LOW_MAX:,} for two"
    if band is Budget.MEDIUM:
        return f"₹{LOW_MAX + 1:,}–₹{MEDIUM_MAX:,} for two"
    return f"over ₹{MEDIUM_MAX:,} for two"
