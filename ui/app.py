"""Streamlit UI: preferences in, recommendation cards out.

Run from the repo root::

    streamlit run ui/app.py

By default the UI calls the recommendation service in-process (one process,
simplest for demos). Set ``RECOMMENDER_API_URL=http://localhost:8000`` to call
the FastAPI service instead.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import httpx  # noqa: E402
import streamlit as st  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from restaurant_recommender.config import get_settings  # noqa: E402
from restaurant_recommender.data.budget import LOW_MAX, MEDIUM_MAX  # noqa: E402
from restaurant_recommender.data.store import DataUnavailableError, RestaurantStore  # noqa: E402
from restaurant_recommender.llm.client import build_client  # noqa: E402
from restaurant_recommender.models.schemas import PreferenceRequest  # noqa: E402
from restaurant_recommender.services.recommend import RecommendationService  # noqa: E402

API_URL = os.getenv("RECOMMENDER_API_URL", "").strip().rstrip("/")
OPTION_LIMIT = 300
BUDGET_LABELS = {
    "any": "Any",
    "low": f"Low · up to ₹{LOW_MAX}",
    "medium": f"Medium · ₹{LOW_MAX + 1}–{MEDIUM_MAX:,}",
    "high": f"High · over ₹{MEDIUM_MAX:,}",
}
FILTER_LABELS = {"location": "location", "cuisine": "cuisine", "budget": "budget", "min_rating": "rating"}
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|$~<>])")


class UIError(Exception):
    """A problem to show the user as-is."""


def md(text: str) -> str:
    """Escape Markdown (and Streamlit's $math$) in data-driven text."""
    return _MD_SPECIAL.sub(r"\\\1", text)


# --------------------------------------------------------------------------- backend access


@st.cache_resource(show_spinner="Loading restaurants…")
def get_service() -> RecommendationService:
    settings = get_settings()
    store = RestaurantStore.from_parquet(settings.data_path)
    return RecommendationService(store, build_client(settings), settings)


@st.cache_data(ttl=600, show_spinner=False)
def get_options() -> dict[str, Any]:
    if API_URL:
        response = httpx.get(f"{API_URL}/api/v1/options", params={"limit": OPTION_LIMIT}, timeout=10)
        response.raise_for_status()
        return response.json()
    service = get_service()
    return {
        "locations": [*service.store.cities(), *service.store.localities()][:OPTION_LIMIT],
        "cuisines": service.store.cuisines()[:OPTION_LIMIT],
    }


def backend_status() -> str:
    if API_URL:
        return f"API at `{API_URL}`"  # code span: escaping would break the auto-linked URL
    service = get_service()
    if service.llm:
        llm = f"{get_settings().llm_provider.title()} · {service.llm.model} · prompt {service.prompt.name}"
    else:
        llm = "LLM not configured (rating fallback)"
    return f"{len(service.store):,} restaurants · {llm}"


def run_query(payload: dict[str, Any]) -> dict[str, Any]:
    if API_URL:
        try:
            response = httpx.post(f"{API_URL}/api/v1/recommendations", json=payload, timeout=60)
        except httpx.HTTPError as exc:
            raise UIError(f"Couldn't reach the recommendation API at {API_URL} ({type(exc).__name__}).") from exc
        if response.status_code == 422:
            details = response.json().get("detail", [])
            messages = [f"{'.'.join(map(str, d.get('loc', [])[1:]))}: {d.get('msg')}" for d in details]
            raise UIError("Please check your inputs: " + "; ".join(messages))
        if response.status_code >= 400:
            raise UIError(f"The API returned {response.status_code}: {response.json().get('detail', response.text)}")
        return response.json()

    try:
        prefs = PreferenceRequest(**payload)
    except ValidationError as exc:
        messages = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
        raise UIError("Please check your inputs: " + "; ".join(messages)) from exc
    return get_service().recommend(prefs).model_dump(mode="json")


# --------------------------------------------------------------------------- rendering


def rating_badge(rating: float | None) -> None:
    if rating is None:
        st.badge("Not rated", color="gray")
        return
    color = "green" if rating >= 4.0 else "blue" if rating >= 3.5 else "orange"
    st.badge(f"{rating:.1f}", icon=":material/star:", color=color)


def render_results(result: dict[str, Any]) -> None:
    recs = result.get("recommendations") or []
    meta = result.get("meta") or {}

    for note in meta.get("query_notes") or []:  # Phase 4: spelling corrections / synonyms
        st.caption(f":material/spellcheck: {md(note)}")

    if not recs:  # UI-01
        st.warning(result.get("summary") or "No restaurants matched your preferences.", icon=":material/search_off:")
        return

    if meta.get("fallback_used"):  # UI-05
        st.caption(f":material/info: {md(meta.get('fallback_reason') or 'AI ranking unavailable')}")
    if result.get("summary"):  # UI-04: hidden when absent
        st.info(md(result["summary"]), icon=":material/auto_awesome:")

    for item in recs:
        with st.container(border=True):
            title, badge = st.columns([6, 1], vertical_alignment="center")
            title.markdown(f"**{item['rank']}. {md(item['name'])}**")
            with badge:
                rating_badge(item.get("rating"))
            details = [item.get("cuisine"), item.get("location"), item.get("rest_type")]
            st.caption(" · ".join(md(d) for d in details if d))
            cost = item.get("estimated_cost") or "N/A"  # UI-07
            band = f" · {item['budget_band']} budget" if item.get("budget_band") else ""
            votes = f" · {item['votes']:,} votes" if item.get("votes") else ""
            st.markdown(f"**{md(cost)}**{band}{votes}")
            st.write(md(item.get("explanation") or ""))

    filters = ", ".join(FILTER_LABELS.get(f, f) for f in meta.get("filters_applied", []))
    total_ms = (meta.get("timings_ms") or {}).get("total")
    st.caption(
        f"{meta.get('candidates_considered', 0)} shortlisted from {meta.get('total_matches', 0):,} matches"
        + (f" · filtered by {filters}" if filters else "")
        + (f" · {total_ms:.0f} ms" if total_ms and total_ms < 1000 else f" · {total_ms / 1000:.1f} s" if total_ms else "")
        + (" · :material/bolt: cached" if meta.get("cache_hit") else "")
        + (f" · ranked by {md(meta['model'])}" if meta.get("model") else "")
    )


# --------------------------------------------------------------------------- page

st.set_page_config(page_title="Restaurant Recommender", page_icon=":material/restaurant:", layout="centered")
st.title("Where should we eat?")
st.caption("Real Zomato listings for Bangalore, filtered by your preferences and ranked by AI.")

try:
    options = get_options()
    status = backend_status()
except DataUnavailableError as exc:  # DI-13
    st.error(str(exc))
    st.code("python -m restaurant_recommender.data.ingest", language="bash")
    st.stop()
except httpx.HTTPError as exc:  # UI-02
    st.error(f"Couldn't reach the recommendation API at {API_URL} ({type(exc).__name__}).")
    st.stop()

locations = options.get("locations") or ["Bangalore"]
cuisines = options.get("cuisines") or []

with st.form("preferences"):
    left, right = st.columns(2)
    location = left.selectbox(
        "Location",
        locations,
        index=0,
        accept_new_options=True,
        help="Bangalore, or one of its localities. Type to search.",
    )
    cuisine = right.selectbox(
        "Cuisine",
        ["Any cuisine", *cuisines],
        index=0,
        accept_new_options=True,
        help="Type to search. Separate several with commas to match any of them.",
    )
    budget = st.segmented_control(
        "Budget (cost for two)",
        list(BUDGET_LABELS),
        default="any",
        required=True,
        format_func=BUDGET_LABELS.get,
    )
    rating_col, count_col = st.columns([3, 2])
    min_rating = rating_col.slider("Minimum rating", 0.0, 5.0, 3.5, 0.1, format="%.1f")
    top_n = count_col.slider("Number of results", 1, 10, 5)
    additional = st.text_input(
        "Anything else?",
        placeholder="family-friendly, quick service, good for groups…",
        max_chars=500,
        help="A soft preference: the AI uses it to rank and explain, but it never overrides the filters above.",
    )
    submitted = st.form_submit_button("Find restaurants", type="primary", width="stretch")

if submitted:
    payload = {
        "location": location or "",
        "cuisine": None if cuisine in (None, "Any cuisine") else cuisine,
        "budget": None if budget in (None, "any") else budget,
        "min_rating": min_rating or None,
        "additional_preferences": additional or "",
        "top_n": top_n,
    }
    # A new submit interrupts the previous run, so only the latest result is shown (UI-06, IN-14).
    with st.spinner("Finding and ranking restaurants…", show_time=True):
        try:
            st.session_state["result"] = run_query(payload)
            st.session_state["error"] = None
        except UIError as exc:
            st.session_state["result"], st.session_state["error"] = None, str(exc)
        except DataUnavailableError as exc:
            st.session_state["result"], st.session_state["error"] = None, str(exc)

if st.session_state.get("error"):
    st.error(st.session_state["error"])
elif st.session_state.get("result"):
    render_results(st.session_state["result"])

st.caption(f":material/database: {status}")  # our own text, not user data
