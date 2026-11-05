"""Pydantic schemas shared by the data layer, the recommendation service, the API and the UI."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from restaurant_recommender.config import MAX_TOP_N, get_settings

MAX_LOCATION_LEN = 100
MAX_CUISINE_LEN = 100
MAX_ADDITIONAL_LEN = 500

_WS = re.compile(r"\s+")
# Values the UI/API may send to mean "no filter on this field".
_ANY_VALUES = {"", "any", "any cuisine", "all", "*"}


def collapse_ws(value: str) -> str:
    """Trim and collapse internal whitespace (edge cases IN-08 / IN-09)."""
    return _WS.sub(" ", value).strip()


class Budget(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# --------------------------------------------------------------------------- request


class PreferenceRequest(BaseModel):
    """User preferences. `location` is required; the other hard filters are optional.

    Hard filters (location, budget, cuisine, min_rating) constrain results.
    `additional_preferences` is a soft signal that only the LLM sees.
    """

    model_config = ConfigDict(extra="forbid")

    location: str = Field(
        min_length=1,
        max_length=MAX_LOCATION_LEN,
        description="City or locality, e.g. 'Bangalore' or 'Indiranagar'. "
        "Comma-separated parts must all match.",
        examples=["Indiranagar"],
    )
    budget: Budget | None = Field(default=None, description="low | medium | high; omit for any.")
    cuisine: str | None = Field(
        default=None,
        max_length=MAX_CUISINE_LEN,
        description="One cuisine, or several separated by commas (any may match). Omit for any.",
        examples=["Italian"],
    )
    min_rating: float | None = Field(
        default=None,
        ge=0,
        le=5,
        allow_inf_nan=False,
        description="Minimum rating out of 5. Omit or 0 for no rating filter.",
        examples=[3.5],
    )
    additional_preferences: str = Field(
        default="",
        max_length=MAX_ADDITIONAL_LEN,
        description="Free text such as 'family-friendly, quick service'. Soft signal only.",
    )
    top_n: int = Field(
        default_factory=lambda: get_settings().top_n_default,
        ge=1,
        le=MAX_TOP_N,
        description=f"Number of recommendations to return (1-{MAX_TOP_N}).",
    )

    @field_validator("location", mode="before")
    @classmethod
    def _clean_location(cls, value: Any) -> Any:
        return collapse_ws(value) if isinstance(value, str) else value

    @field_validator("cuisine", mode="before")
    @classmethod
    def _clean_cuisine(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            value = collapse_ws(value)
            return None if value.lower() in _ANY_VALUES else value
        return value

    @field_validator("budget", mode="before")
    @classmethod
    def _clean_budget(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip().lower()
            return None if value in _ANY_VALUES else value
        return value

    @field_validator("additional_preferences", mode="before")
    @classmethod
    def _clean_additional(cls, value: Any) -> Any:
        if value is None:  # edge case IN-13: null == omitted
            return ""
        return collapse_ws(value) if isinstance(value, str) else value


# --------------------------------------------------------------------------- restaurant


class Restaurant(BaseModel):
    """Canonical restaurant record (one row of data/processed/restaurants.parquet)."""

    id: str
    name: str
    city: str
    locality: str | None = None
    areas: list[str] = Field(default_factory=list, description="Zomato listing zones")
    address: str | None = None
    cuisines: list[str] = Field(default_factory=list)
    rating: float | None = None
    votes: int = 0
    cost_for_two: int | None = None
    budget_band: Budget | None = None
    rest_type: str | None = None
    online_order: bool | None = None
    book_table: bool | None = None
    dish_liked: str | None = None
    listed_types: list[str] = Field(default_factory=list)

    @property
    def cuisine_display(self) -> str:
        return ", ".join(self.cuisines) if self.cuisines else "N/A"

    @property
    def location_display(self) -> str:
        parts = [p for p in (self.locality, self.city) if p]
        return ", ".join(dict.fromkeys(parts)) or "N/A"


# --------------------------------------------------------------------------- response


class RecommendationItem(BaseModel):
    """One recommendation card. Facts come from the dataset; rank/explanation from the LLM."""

    rank: int
    source_id: str
    name: str
    cuisine: str
    rating: float | None
    votes: int
    estimated_cost: str = Field(description="Display string, e.g. '₹1,200 for two' or 'N/A'")
    cost_for_two: int | None
    budget_band: Budget | None
    location: str
    rest_type: str | None = None
    explanation: str
    explanation_source: Literal["llm", "template"]


class RecommendationMeta(BaseModel):
    request_id: str
    candidates_considered: int = Field(description="Shortlist size sent to ranking (after cap)")
    total_matches: int = Field(description="Restaurants matching all hard filters (before cap)")
    filters_applied: list[str]
    filter_counts: dict[str, int] = Field(
        description="Restaurants remaining after each filter, in the order applied"
    )
    llm_used: bool
    fallback_used: bool
    fallback_reason: str | None = None
    backfilled: int = Field(0, description="Slots filled from rating order after the LLM ranking")
    model: str | None = None
    prompt_variant: str | None = None
    cache_hit: bool = Field(False, description="Served from the short-lived response cache")
    query_notes: list[str] = Field(
        default_factory=list,
        description="How the query was interpreted, e.g. spelling corrections or cuisine synonyms",
    )
    usage: dict[str, int] = Field(default_factory=dict, description="LLM token usage for this request")
    timings_ms: dict[str, float] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    summary: str | None
    recommendations: list[RecommendationItem]
    meta: RecommendationMeta
