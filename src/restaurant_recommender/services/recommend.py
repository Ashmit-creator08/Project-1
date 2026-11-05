"""Orchestration: preferences -> (cache) -> filter -> LLM rank/explain -> fact merge -> response.

Invariants (edge-case doc §8) enforced here:

* every returned `source_id` is in this request's filtered shortlist (INV-01);
* name / cuisine / rating / cost / location always come from the store (INV-02);
* result count <= min(top_n, candidates_considered) (INV-03);
* no LLM call when nothing matched (INV-04);
* additional preferences never change the hard-filtered set (INV-05).
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field

from restaurant_recommender.config import Settings
from restaurant_recommender.data.filter import FilterResult, filter_restaurants
from restaurant_recommender.data.store import RestaurantStore
from restaurant_recommender.llm.client import ChatClient, LLMResult, as_result
from restaurant_recommender.llm.errors import LLMError, LLMParseError
from restaurant_recommender.llm.parser import ParsedRanking, parse_ranking
from restaurant_recommender.llm.prompts import build_messages, build_repair_messages, get_variant, ranking_schema
from restaurant_recommender.models.schemas import (
    PreferenceRequest,
    RecommendationItem,
    RecommendationMeta,
    RecommendationResponse,
    Restaurant,
)
from restaurant_recommender.services.cache import TTLCache, cache_key
from restaurant_recommender.services.explain import (
    empty_summary,
    fallback_summary,
    format_cost,
    ranked_summary,
    template_explanation,
)
from restaurant_recommender.services.telemetry import Telemetry

log = logging.getLogger(__name__)

NOT_CONFIGURED_REASON = "No LLM configured (set GROQ_API_KEY); ranked by rating"


@dataclass
class _LLMStats:
    calls: int = 0
    repaired: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    rate_limit: dict[str, str] = field(default_factory=dict)

    def add(self, result: LLMResult) -> None:
        self.calls += 1
        self.prompt_tokens += result.prompt_tokens or 0
        self.completion_tokens += result.completion_tokens or 0
        if result.rate_limit:
            self.rate_limit = result.rate_limit

    @property
    def usage(self) -> dict[str, int]:
        if not (self.prompt_tokens or self.completion_tokens):
            return {}
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}


class RecommendationService:
    def __init__(
        self,
        store: RestaurantStore,
        llm: ChatClient | None,
        settings: Settings,
        *,
        telemetry: Telemetry | None = None,
        cache: TTLCache[RecommendationResponse] | None = None,
        prompt_variant: str | None = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.settings = settings
        self.prompt = get_variant(prompt_variant or settings.prompt_variant)
        self.telemetry = telemetry or Telemetry()
        if cache is None and settings.cache_ttl_seconds > 0:
            cache = TTLCache(settings.cache_max_entries, settings.cache_ttl_seconds)
        self.cache = cache

    @property
    def model_name(self) -> str | None:
        return getattr(self.llm, "model", None) if self.llm is not None else None

    def recommend(self, prefs: PreferenceRequest, request_id: str | None = None) -> RecommendationResponse:
        request_id = request_id or uuid.uuid4().hex[:12]
        started = time.perf_counter()

        key = None
        if self.cache is not None:
            key = cache_key(
                prefs, data_version=self.store.version, prompt_variant=self.prompt.name, model=self.model_name
            )
            cached = self.cache.get(key)
            if cached is not None:
                return self._from_cache(cached, request_id, started)

        response, stats, fallback_kind, dropped = self._compute(prefs, request_id, started)

        if key is not None and not response.meta.fallback_used:
            self.cache.set(key, response.model_copy(deep=True))
        meta = response.meta
        self.telemetry.record_request(
            timings_ms=meta.timings_ms,
            empty=not response.recommendations,
            cache_hit=False,
            llm_used=meta.llm_used,
            fallback_kind=fallback_kind,
            llm_calls=stats.calls,
            repaired=stats.repaired,
            dropped_ids=dropped,
            backfilled=meta.backfilled,
            prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.completion_tokens,
            rate_limit=stats.rate_limit,
        )
        return response

    # ----------------------------------------------------------------- pipeline

    def _compute(
        self, prefs: PreferenceRequest, request_id: str, started: float
    ) -> tuple[RecommendationResponse, _LLMStats, str | None, int]:
        stats = _LLMStats()
        result = filter_restaurants(self.store, prefs, self.settings.max_candidates)
        timings = {"filter": round((time.perf_counter() - started) * 1000, 1)}
        candidates = result.candidates

        if not candidates:  # INV-04: skip the LLM entirely
            timings["total"] = round((time.perf_counter() - started) * 1000, 1)
            self._log(request_id, result, llm_ms=None, fallback=None)
            response = RecommendationResponse(
                summary=empty_summary(prefs, result),
                recommendations=[],
                meta=self._meta(request_id, result, timings, llm_used=False, fallback_reason=None),
            )
            return response, stats, None, 0

        target = min(prefs.top_n, len(candidates))
        ranking: ParsedRanking | None = None
        fallback_reason: str | None = None
        fallback_kind: str | None = None
        llm_ms: float | None = None

        if self.llm is None:
            fallback_reason, fallback_kind = NOT_CONFIGURED_REASON, "not_configured"
        else:
            llm_started = time.perf_counter()
            try:
                ranking = self._rank_with_llm(prefs, result, target, request_id, stats)
            except LLMError as exc:
                fallback_reason, fallback_kind = str(exc), exc.kind
                log.warning("request_id=%s LLM failed (%s), using rating fallback: %s", request_id, exc.kind, exc)
            llm_ms = (time.perf_counter() - llm_started) * 1000
            timings["llm"] = round(llm_ms, 1)

        items, backfilled = self._merge(ranking, candidates, prefs, target, result.cuisine_terms)
        if ranking is None:
            summary = fallback_summary(len(items), prefs, configured=self.llm is not None)
        elif ranking.summary and self._summary_is_consistent(ranking.summary, items, candidates):
            summary = ranking.summary
        else:
            summary = ranked_summary(len(items), self._prompt_prefs(prefs, result))

        timings["total"] = round((time.perf_counter() - started) * 1000, 1)
        self._log(request_id, result, llm_ms=llm_ms, fallback=fallback_reason)
        response = RecommendationResponse(
            summary=summary,
            recommendations=items,
            meta=self._meta(
                request_id,
                result,
                timings,
                llm_used=ranking is not None,
                fallback_reason=fallback_reason,
                backfilled=backfilled,
                usage=stats.usage,
            ),
        )
        dropped = len(ranking.dropped_ids) if ranking is not None else 0
        return response, stats, fallback_kind, dropped

    def _from_cache(
        self, cached: RecommendationResponse, request_id: str, started: float
    ) -> RecommendationResponse:
        response = cached.model_copy(deep=True)
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        response.meta.request_id = request_id
        response.meta.cache_hit = True
        response.meta.usage = {}
        response.meta.timings_ms = {"total": elapsed}
        self.telemetry.record_request(
            timings_ms=response.meta.timings_ms,
            empty=not response.recommendations,
            cache_hit=True,
            llm_used=False,
            fallback_kind=None,
        )
        log.info("request_id=%s served from cache in %.1f ms", request_id, elapsed)
        return response

    # ----------------------------------------------------------------- LLM

    @staticmethod
    def _prompt_prefs(prefs: PreferenceRequest, result: FilterResult) -> PreferenceRequest:
        """Show the model (and summaries) the corrected location, e.g. 'Koramangala' not 'Koramangla'."""
        if result.location_label and result.location_label != prefs.location:
            return prefs.model_copy(update={"location": result.location_label})
        return prefs

    def _call(self, messages, schema: dict, stats: _LLMStats) -> LLMResult:
        assert self.llm is not None
        result = as_result(self.llm.complete(messages, schema=schema))
        stats.add(result)
        return result

    def _rank_with_llm(
        self,
        prefs: PreferenceRequest,
        result: FilterResult,
        target: int,
        request_id: str,
        stats: _LLMStats,
    ) -> ParsedRanking:
        messages, included = build_messages(
            self._prompt_prefs(prefs, result), result.candidates, target, variant=self.prompt
        )
        allowed = {c.id for c in included}
        schema = ranking_schema([c.id for c in included])
        bad_output = ""
        try:
            reply = self._call(messages, schema, stats)
            bad_output = reply.text
            ranking = parse_ranking(reply.text, allowed, target)
        except LLMParseError as exc:
            # One repair attempt, then the caller falls back (LM-05/LM-06).
            log.info("request_id=%s unusable LLM reply (%s); asking for a repair", request_id, exc)
            stats.repaired = True
            reply = self._call(build_repair_messages(messages, exc.raw or bad_output, str(exc)), schema, stats)
            ranking = parse_ranking(reply.text, allowed, target)
        if ranking.dropped_ids:
            log.warning(
                "request_id=%s dropped %d unknown ids from LLM output", request_id, len(ranking.dropped_ids)
            )
        return ranking

    # ----------------------------------------------------------------- merge

    def _merge(
        self,
        ranking: ParsedRanking | None,
        candidates: list[Restaurant],
        prefs: PreferenceRequest,
        target: int,
        cuisine_terms_used: list[str] | None = None,
    ) -> tuple[list[RecommendationItem], int]:
        by_id = {c.id: c for c in candidates}
        chosen: list[tuple[Restaurant, str, str]] = []
        used: set[str] = set()

        def template(restaurant: Restaurant) -> str:
            return template_explanation(restaurant, prefs, cuisine_terms_used)

        if ranking is not None:
            for rid, explanation in ranking.items:
                restaurant = by_id.get(rid)
                if restaurant is None or rid in used:
                    continue
                used.add(rid)
                if explanation:
                    chosen.append((restaurant, explanation, "llm"))
                else:  # LM-14: never return an empty explanation
                    chosen.append((restaurant, template(restaurant), "template"))

        llm_count = len(chosen)
        for restaurant in candidates:  # already rating-sorted (fallback / LM-08 / LM-10 backfill)
            if len(chosen) >= target:
                break
            if restaurant.id not in used:
                used.add(restaurant.id)
                chosen.append((restaurant, template(restaurant), "template"))

        items = [
            RecommendationItem(
                rank=rank,
                source_id=r.id,
                name=r.name,
                cuisine=r.cuisine_display,
                rating=r.rating,
                votes=r.votes,
                estimated_cost=format_cost(r.cost_for_two),
                cost_for_two=r.cost_for_two,
                budget_band=r.budget_band,
                location=r.location_display,
                rest_type=r.rest_type,
                explanation=explanation,
                explanation_source=source,
            )
            for rank, (r, explanation, source) in enumerate(chosen[:target], start=1)
        ]
        backfilled = max(0, len(items) - llm_count) if ranking is not None else 0
        return items, backfilled

    @staticmethod
    def _summary_is_consistent(
        summary: str, items: list[RecommendationItem], candidates: list[Restaurant]
    ) -> bool:
        """Reject an LLM summary that names a shortlisted restaurant we did not return (LM-16)."""
        returned = {i.name.lower() for i in items}
        text = summary.lower()
        for c in candidates:
            name = c.name.lower()
            if name in returned or len(name) < 4:
                continue
            if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text):
                return False
        return True

    # ----------------------------------------------------------------- meta / logs

    def _meta(
        self,
        request_id: str,
        result: FilterResult,
        timings: dict[str, float],
        *,
        llm_used: bool,
        fallback_reason: str | None,
        backfilled: int = 0,
        usage: dict[str, int] | None = None,
    ) -> RecommendationMeta:
        return RecommendationMeta(
            request_id=request_id,
            candidates_considered=len(result.candidates),
            total_matches=result.total_matches,
            filters_applied=result.filters_applied,
            filter_counts=result.filter_counts,
            llm_used=llm_used,
            fallback_used=fallback_reason is not None,
            fallback_reason=fallback_reason,
            backfilled=backfilled,
            model=self.model_name if llm_used else None,
            prompt_variant=self.prompt.name if llm_used else None,
            query_notes=list(result.notes),
            usage=usage or {},
            timings_ms=timings,
        )

    @staticmethod
    def _log(request_id: str, result: FilterResult, *, llm_ms: float | None, fallback: str | None) -> None:
        # Counts and timings only: no user free text, no secrets (CF-04, CF-05).
        log.info(
            "request_id=%s filters=%s counts=%s shortlist=%d llm_ms=%s fallback=%s",
            request_id,
            ",".join(result.filters_applied),
            result.filter_counts,
            len(result.candidates),
            f"{llm_ms:.0f}" if llm_ms is not None else "-",
            "yes" if fallback else "no",
        )
