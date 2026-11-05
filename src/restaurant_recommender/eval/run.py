"""Eval harness CLI: runs the eval set and writes a scorecard (docs/eval.md).

Examples::

    # Live Groq run (uses GROQ_API_KEY), LLM-as-judge on the judged cases
    python -m restaurant_recommender.eval.run

    # Prompt A/B: every variant, side by side, winner recorded
    python -m restaurant_recommender.eval.run --prompt all

    # Offline / CI: mocked LLM, hard metrics only
    python -m restaurant_recommender.eval.run --llm mock

    # Failure injection only
    python -m restaurant_recommender.eval.run --inject llm_failures

Exit code is 1 when any hard metric or invariant fails.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from restaurant_recommender.config import REPO_ROOT, Settings, get_settings
from restaurant_recommender.data.filter import filter_restaurants
from restaurant_recommender.data.store import DataUnavailableError, RestaurantStore
from restaurant_recommender.eval.checks import CaseCheck, check_response
from restaurant_recommender.eval.judge import HeuristicJudge, JudgeScore, LLMJudge, mentions_preference
from restaurant_recommender.eval.mock import OfflineRanker, ScriptedLLM, hallucinating
from restaurant_recommender.llm.client import ChatClient, build_client
from restaurant_recommender.llm.errors import LLMRequestError, LLMTimeoutError
from restaurant_recommender.llm.prompts import VARIANTS, build_messages
from restaurant_recommender.models.schemas import PreferenceRequest, RecommendationResponse
from restaurant_recommender.services.recommend import RecommendationService
from restaurant_recommender.services.telemetry import Telemetry, percentile

DEFAULT_CASES = REPO_ROOT / "tests" / "eval" / "cases.json"
DEFAULT_OUT = REPO_ROOT / "evals" / "scorecard-latest.md"
TOKEN_RESERVE = 6000  # pause when Groq's remaining TPM budget drops below this


# --------------------------------------------------------------------------- plumbing


class CountingClient:
    """Wraps a client to count calls per case (empty-path check: INV-04)."""

    def __init__(self, inner: ChatClient) -> None:
        self.inner = inner
        self.model = inner.model
        self.calls = 0

    def complete(self, messages, *, schema: dict | None = None):
        self.calls += 1
        return self.inner.complete(messages, schema=schema)


def parse_duration(text: str | None) -> float:
    """Groq reset headers: '7.66s', '2m59.56s', '120ms' -> seconds."""
    if not text:
        return 0.0
    total = 0.0
    for value, unit in re.findall(r"([\d.]+)(ms|m|s|h)", text):
        total += float(value) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total


class Throttle:
    def __init__(self, telemetry: Telemetry, delay: float, enabled: bool) -> None:
        self.telemetry = telemetry
        self.delay = delay
        self.enabled = enabled
        self.waited = 0.0

    def after_llm_call(self, headroom: dict[str, str] | None = None) -> None:
        """Pause after a live call; longer when the model's TPM budget is nearly used up.

        `headroom` defaults to the ranker's latest rate-limit headers (from telemetry);
        pass the judge's headers after a judge call.
        """
        if not self.enabled:
            return
        pause = self.delay
        if headroom is None:
            headroom = self.telemetry.snapshot()["llm"]["rate_limit_remaining"]
        raw = headroom.get("x-ratelimit-remaining-tokens")
        try:
            remaining = int(float(raw)) if raw is not None else None
        except (ValueError, OverflowError):
            remaining = None
        if remaining is not None and remaining < TOKEN_RESERVE:
            pause = max(pause, min(parse_duration(headroom.get("x-ratelimit-reset-tokens")) + 0.5, 65.0))
        if pause > 0:
            self.waited += pause
            time.sleep(pause)


@dataclass
class CaseOutcome:
    case_id: str
    description: str
    tags: list[str]
    passed: bool = True
    failures: list[str] = field(default_factory=list)
    check: CaseCheck | None = None
    response: RecommendationResponse | None = None
    shortlist_ids: list[str] = field(default_factory=list)
    llm_calls: int = 0
    seconds: float = 0.0
    judge_scores: list[JudgeScore] = field(default_factory=list)
    heuristic_scores: list[JudgeScore] = field(default_factory=list)
    coverage: list[bool] = field(default_factory=list)
    summary_consistent: bool | None = None


@dataclass
class VariantReport:
    variant: str
    outcomes: list[CaseOutcome] = field(default_factory=list)
    injections: list[tuple[str, bool, str]] = field(default_factory=list)
    latency_s: list[float] = field(default_factory=list)
    prompt_chars: int = 0
    telemetry: Telemetry = field(default_factory=Telemetry)

    # ------------------------------------------------------------- metrics

    def _items(self, attr: str) -> tuple[int, int]:
        checks = [o.check for o in self.outcomes if o.check is not None]
        return sum(getattr(c, attr) for c in checks), sum(c.items for c in checks)

    def ratio(self, attr: str) -> float | None:
        num, den = self._items(attr)
        return num / den if den else None

    @property
    def cardinality(self) -> float | None:
        checks = [o.check for o in self.outcomes if o.check is not None]
        return sum(c.cardinality_ok for c in checks) / len(checks) if checks else None

    @property
    def duplicate_free(self) -> float | None:
        checks = [o.check for o in self.outcomes if o.check is not None]
        return sum(c.duplicate_free for c in checks) / len(checks) if checks else None

    @property
    def empty_path(self) -> tuple[int, int]:
        empties = [o for o in self.outcomes if o.response is not None and o.response.meta.candidates_considered == 0]
        return sum(o.llm_calls == 0 for o in empties), len(empties)

    @property
    def hard_pass(self) -> bool:
        metrics = [self.ratio("id_valid"), self.ratio("constraint_ok"), self.ratio("fact_ok"), self.ratio("complete"),
                   self.cardinality, self.duplicate_free]
        ok_empty, n_empty = self.empty_path
        return (
            all(m is None or m == 1.0 for m in metrics)
            and ok_empty == n_empty
            and all(passed for _, passed, _ in self.injections)
            and all(o.passed for o in self.outcomes)
        )

    def acceptable(self, which: str = "judge") -> float | None:
        scores = [s for o in self.outcomes for s in (o.judge_scores if which == "judge" else o.heuristic_scores)]
        return sum(s.acceptable for s in scores) / len(scores) if scores else None

    def mean_total(self, which: str = "judge") -> float | None:
        scores = [s for o in self.outcomes for s in (o.judge_scores if which == "judge" else o.heuristic_scores)]
        return sum(s.total for s in scores) / len(scores) if scores else None

    @property
    def coverage(self) -> float | None:
        flags = [f for o in self.outcomes for f in o.coverage]
        return sum(flags) / len(flags) if flags else None

    @property
    def summary_consistency(self) -> float | None:
        flags = [o.summary_consistent for o in self.outcomes if o.summary_consistent is not None]
        return sum(flags) / len(flags) if flags else None

    @property
    def llm_fallback_rate(self) -> float | None:
        live = [o for o in self.outcomes if o.llm_calls > 0 and o.response is not None and "inject" not in o.tags]
        return sum(o.response.meta.fallback_used for o in live) / len(live) if live else None


# --------------------------------------------------------------------------- running


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_expectations(case: dict, outcome: CaseOutcome, prefs: PreferenceRequest, by_id: dict[str, CaseOutcome], max_candidates: int) -> None:
    expect = case.get("expect", {})
    response = outcome.response
    recs, meta = response.recommendations, response.meta

    def fail(message: str) -> None:
        outcome.failures.append(message)

    if "non_empty" in expect and bool(recs) is not expect["non_empty"]:
        fail(f"expected non_empty={expect['non_empty']}")
    if "llm_called" in expect and (outcome.llm_calls > 0) is not expect["llm_called"]:
        fail(f"expected llm_called={expect['llm_called']}, calls={outcome.llm_calls}")
    if "fallback" in expect and meta.fallback_used is not expect["fallback"]:
        fail(f"expected fallback={expect['fallback']} ({meta.fallback_reason})")
    if expect.get("capped") and not (meta.total_matches > max_candidates == meta.candidates_considered):
        fail("expected the shortlist to be capped")
    if expect.get("fewer_than_top_n") and not (0 < len(recs) < prefs.top_n):
        fail("expected fewer items than top_n")
    if "count" in expect and len(recs) != expect["count"]:
        fail(f"expected {expect['count']} items, got {len(recs)}")
    if "notes_contain" in expect and not any(expect["notes_contain"].lower() in n.lower() for n in meta.query_notes):
        fail(f"expected a query note containing {expect['notes_contain']!r}, got {meta.query_notes}")
    if "same_as" in expect and expect["same_as"] in by_id:
        if outcome.shortlist_ids != by_id[expect["same_as"]].shortlist_ids:
            fail(f"shortlist differs from {expect['same_as']}")


def injected_client(kind: str) -> ScriptedLLM:
    return {
        "llm_failure": lambda: ScriptedLLM(LLMTimeoutError("LLM did not respond within 20s")),
        "invalid_json": lambda: ScriptedLLM("not json", "still not json"),
        "hallucinated": lambda: ScriptedLLM(hallucinating),
    }[kind]()


def run_cases(
    cases: list[dict],
    store: RestaurantStore,
    settings: Settings,
    variant: str,
    base_client: ChatClient,
    judge: LLMJudge | None,
    throttle: Throttle,
    report: VariantReport,
    live: bool,
) -> None:
    heuristic = HeuristicJudge()
    by_id: dict[str, CaseOutcome] = {}
    for case in cases:
        outcome = CaseOutcome(case["id"], case.get("description", ""), list(case.get("tags", [])))
        expect = case.get("expect", {})
        if expect.get("invalid"):
            try:
                PreferenceRequest(**case["prefs"])
                outcome.failures.append("invalid preferences were accepted")
            except ValidationError:
                pass
            outcome.passed = not outcome.failures
            report.outcomes.append(outcome)
            continue

        prefs = PreferenceRequest(**case["prefs"])
        inject = case.get("inject")
        inner = injected_client(inject) if inject else base_client
        if inject:
            outcome.tags.append("inject")
        client = CountingClient(inner)
        service = RecommendationService(store, client, settings, telemetry=report.telemetry, prompt_variant=variant)
        started = time.perf_counter()
        outcome.response = service.recommend(prefs, request_id=f"eval-{case['id']}")
        outcome.seconds = time.perf_counter() - started
        outcome.llm_calls = client.calls
        outcome.shortlist_ids = [c.id for c in filter_restaurants(store, prefs, settings.max_candidates).candidates]
        outcome.check = check_response(store, prefs, outcome.response, settings.max_candidates)
        outcome.failures += outcome.check.violations
        _check_expectations(case, outcome, prefs, by_id, settings.max_candidates)

        response = outcome.response
        if live and not inject and response.meta.llm_used:
            shortlist = filter_restaurants(store, prefs, settings.max_candidates).candidates
            outcome.summary_consistent = bool(response.summary) and RecommendationService._summary_is_consistent(
                response.summary, response.recommendations, shortlist
            )
            for item in response.recommendations:
                if item.explanation_source != "llm":
                    continue
                fact = store.get(item.source_id)
                outcome.heuristic_scores.append(heuristic.score(prefs, fact, item.explanation))
                outcome.coverage.append(mentions_preference(item.explanation, prefs, fact))
                if judge is not None and "judge" in outcome.tags:
                    score = judge.score(prefs, fact, item.explanation)
                    if score is not None:
                        outcome.judge_scores.append(score)
                    throttle.after_llm_call(judge.rate_limit)
        if outcome.llm_calls and not inject:
            throttle.after_llm_call()

        outcome.passed = not outcome.failures
        by_id[case["id"]] = outcome
        report.outcomes.append(outcome)
        status = "PASS" if outcome.passed else "FAIL"
        print(f"  [{variant}] {case['id']:<7} {status}  items={len(response.recommendations)} "
              f"llm_calls={outcome.llm_calls} fallback={response.meta.fallback_used} {outcome.seconds:.2f}s", flush=True)


def run_injections(store: RestaurantStore, settings: Settings, variant: str) -> list[tuple[str, bool, str]]:
    """eval.md §B: failure injection with a mocked LLM."""
    prefs = PreferenceRequest(location="Bangalore", cuisine="Italian", budget="medium", min_rating=3.5, top_n=5)
    scenarios = {
        "timeout": (ScriptedLLM(LLMTimeoutError("LLM did not respond within 20s")), "fallback"),
        "http_500": (ScriptedLLM(LLMRequestError("LLM provider returned HTTP 500", kind="http")), "fallback"),
        "rate_limit_429": (ScriptedLLM(LLMRequestError("LLM provider rate limit reached", kind="rate_limit")), "fallback"),
        "invalid_json_twice": (ScriptedLLM("not json", "still not json"), "fallback"),
        "invalid_json_repaired": (ScriptedLLM("not json"), "repaired"),
        "hallucinated_ids": (ScriptedLLM(hallucinating), "backfilled"),
    }
    results = []
    for name, (llm, expectation) in scenarios.items():
        client = CountingClient(llm)
        service = RecommendationService(store, client, settings, prompt_variant=variant)
        response = service.recommend(prefs)
        check = check_response(store, prefs, response, settings.max_candidates)
        meta = response.meta
        ok = check.passed and bool(response.recommendations)
        if expectation == "fallback":
            ok &= meta.fallback_used and all(r.explanation_source == "template" for r in response.recommendations)
        elif expectation == "repaired":
            ok &= meta.llm_used and client.calls == 2
        elif expectation == "backfilled":
            ok &= (not meta.fallback_used) and meta.backfilled > 0
        results.append((name, ok, f"calls={client.calls} fallback={meta.fallback_used} backfilled={meta.backfilled}"))

    empty = CountingClient(OfflineRanker())
    service = RecommendationService(store, empty, settings, prompt_variant=variant)
    response = service.recommend(PreferenceRequest(location="Atlantis", cuisine="Italian"))
    results.append(("empty_candidates_no_call", empty.calls == 0 and not response.recommendations, f"calls={empty.calls}"))
    return results


def latency_sample(store, settings, variant, client, throttle, report, n: int) -> None:
    prefs = PreferenceRequest(location="Bangalore", cuisine="Italian", budget="medium", min_rating=3.5, top_n=5)
    for i in range(n):
        service = RecommendationService(store, client, settings, telemetry=report.telemetry, prompt_variant=variant)
        started = time.perf_counter()
        response = service.recommend(prefs, request_id=f"latency-{i}")
        report.latency_s.append(time.perf_counter() - started)
        if response.meta.llm_used or response.meta.fallback_used:
            throttle.after_llm_call()


def prompt_size(store: RestaurantStore, settings: Settings, variant: str) -> int:
    prefs = PreferenceRequest(location="Bangalore", cuisine="Italian", budget="medium", min_rating=3.5,
                              additional_preferences="family-friendly", top_n=5)
    shortlist = filter_restaurants(store, prefs, settings.max_candidates).candidates
    messages, _ = build_messages(prefs, shortlist, 5, variant=variant)
    return sum(len(m["content"]) for m in messages)


# --------------------------------------------------------------------------- scorecard


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def git_revision() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return f"{sha or 'unknown'}{' (uncommitted changes)' if dirty else ''}"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def choose_winner(reports: list[VariantReport]) -> tuple[str | None, str]:
    eligible = [r for r in reports if r.hard_pass]
    if not eligible:
        return None, "no variant passed every hard metric"
    if len(eligible) == 1:
        return eligible[0].variant, "only variant passing every hard metric"

    def key(r: VariantReport):
        judged = r.acceptable("judge")
        heuristic = r.acceptable("heuristic")
        tokens = r.telemetry.snapshot()["llm"]["prompt_tokens"] or r.prompt_chars
        p95 = percentile(r.latency_s, 95) or 0
        return (-(judged if judged is not None else -1), -(heuristic if heuristic is not None else -1),
                -(r.mean_total("judge") or 0), tokens, p95)

    best = sorted(eligible, key=key)[0]
    return best.variant, ("higher judge Acceptable% (then heuristic Acceptable%, mean score, "
                          "fewer prompt tokens, lower p95 latency)")


def render(reports: list[VariantReport], header: dict[str, str], judged_cases: list[str], injection_only: bool) -> str:
    names = [r.variant for r in reports]
    lines = ["# Eval scorecard", "", "```text"]
    lines += [f"{k + ':':<14}{v}" for k, v in header.items()]
    lines += ["```", ""]

    def row(label: str, values: list[str], target: str = "") -> str:
        return f"| {label} | " + " | ".join(values) + f" | {target} |"

    table_head = ["| Metric | " + " | ".join(names) + " | Target |", "|---|" + "---|" * len(names) + "---|"]

    lines += ["## Hard metrics", "", *table_head]
    if not injection_only:
        lines.append(row("Constraint adherence", [_pct(r.ratio("constraint_ok")) for r in reports], "100%"))
        lines.append(row("ID validity", [_pct(r.ratio("id_valid")) for r in reports], "100%"))
        lines.append(row("Fact fidelity", [_pct(r.ratio("fact_ok")) for r in reports], "100%"))
        lines.append(row("Schema completeness", [_pct(r.ratio("complete")) for r in reports], "100%"))
        lines.append(row("Result cardinality", [_pct(r.cardinality) for r in reports], "100%"))
        lines.append(row("Duplicate-free ranking", [_pct(r.duplicate_free) for r in reports], "100%"))
        lines.append(row("Empty-path correctness", [f"{a}/{b}" for a, b in (r.empty_path for r in reports)], "all"))
    lines.append(row("Fallback suite", [
        f"{'PASS' if all(p for _, p, _ in r.injections) else 'FAIL'} ({sum(p for _, p, _ in r.injections)}/{len(r.injections)})"
        for r in reports
    ], "PASS"))
    lines.append(row("Invariants (INV-01..05)", ["PASS" if r.hard_pass else "FAIL" for r in reports], "PASS"))
    lines.append("")

    if not injection_only:
        lines += ["## Soft metrics", "", *table_head]
        lines.append(row(f"Explanation Acceptable% (LLM judge, {', '.join(judged_cases) or 'none'})",
                         [_pct(r.acceptable("judge")) for r in reports], "≥ 80%"))
        lines.append(row("Mean judge score (0-6)",
                         ["n/a" if r.mean_total("judge") is None else f"{r.mean_total('judge'):.2f}" for r in reports]))
        lines.append(row("Explanation Acceptable% (heuristic, all LLM items)",
                         [_pct(r.acceptable("heuristic")) for r in reports], "≥ 80%"))
        lines.append(row("Preference coverage", [_pct(r.coverage) for r in reports], "≥ 80%"))
        lines.append(row("Summary consistency", [_pct(r.summary_consistency) for r in reports], "≥ 90%"))
        lines.append(row("LLM fallback rate (live cases)", [_pct(r.llm_fallback_rate) for r in reports], "monitor"))
        lines.append(row("e2e latency p50 / p95 (s)", [
            "n/a" if not r.latency_s else f"{percentile(r.latency_s, 50):.2f} / {percentile(r.latency_s, 95):.2f}"
            for r in reports
        ], "p95 < 8 s"))
        lines.append(row("Prompt size, EV-01 shortlist (chars)", [f"{r.prompt_chars:,}" for r in reports]))
        lines.append(row("LLM tokens (prompt / completion, ranker)", [
            f"{r.telemetry.snapshot()['llm']['prompt_tokens']:,} / {r.telemetry.snapshot()['llm']['completion_tokens']:,}"
            for r in reports
        ]))
        lines.append("")

    if len(reports) > 1:
        winner, reason = choose_winner(reports)
        lines += ["## Prompt A/B", "", f"- **Winner:** `{winner}`" if winner else "- **Winner:** none",
                  f"- Rule: hard metrics must be 100% for every variant considered; then {reason}.",
                  f"- Recorded: {header.get('Eval run', '')}", ""]

    for report in reports:
        if not injection_only:
            lines += [f"## Cases — `{report.variant}`", "", "| Case | Result | Items | LLM calls | Fallback | Notes |",
                      "|---|---|---|---|---|---|"]
            for o in report.outcomes:
                resp = o.response
                notes = "; ".join(o.failures) if o.failures else "; ".join(resp.meta.query_notes) if resp else ""
                lines.append(
                    f"| {o.case_id} | {'PASS' if o.passed else 'FAIL'} | "
                    f"{len(resp.recommendations) if resp else '-'} | {o.llm_calls} | "
                    f"{resp.meta.fallback_used if resp else '-'} | {notes.replace('|', '/')} |"
                )
            lines.append("")
        lines += [f"### Failure injection — `{report.variant}`", "", "| Scenario | Result | Detail |", "|---|---|---|"]
        lines += [f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |" for name, ok, detail in report.injections]
        lines.append("")

    failed = [f"{r.variant}:{o.case_id} ({'; '.join(o.failures)})" for r in reports for o in r.outcomes if not o.passed]
    failed += [f"{r.variant}:{name}" for r in reports for name, ok, _ in r.injections if not ok]
    lines += [f"**Cases failed:** {', '.join(failed) if failed else 'none'}  ",
              f"**Decision:** {'Ship' if all(r.hard_pass for r in reports) else 'Fix'}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the eval set and write a scorecard.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--llm", choices=["auto", "live", "mock"], default="auto",
                        help="live = configured provider (Groq); mock = offline ranker (default: live if a key is set)")
    parser.add_argument("--prompt", default=None, help=f"prompt variant or 'all' ({', '.join(VARIANTS)})")
    parser.add_argument("--judge", choices=["auto", "llm", "heuristic", "none"], default="auto")
    parser.add_argument("--inject", choices=["none", "llm_failures"], default="none",
                        help="llm_failures = run only the failure-injection suite")
    parser.add_argument("--latency", type=int, default=None, help="latency samples per variant (default 3 live, 10 mock)")
    parser.add_argument("--delay", type=float, default=None, help="seconds between live LLM calls (default 1.0)")
    parser.add_argument("--tags", default=None,
                        help="comma-separated tags; run only cases with any of them (saves live tokens)")
    args = parser.parse_args(argv)

    settings = get_settings().model_copy(update={"cache_ttl_seconds": 0})  # never measure the cache
    try:
        store = RestaurantStore.from_parquet(settings.data_path)
    except DataUnavailableError as exc:
        print(exc, file=sys.stderr)
        return 2

    live = args.llm == "live" or (args.llm == "auto" and settings.llm_enabled)
    if live and not settings.llm_enabled:
        print("--llm live needs GROQ_API_KEY (or LLM_API_KEY) in .env", file=sys.stderr)
        return 2
    # Eval may wait out a 429 instead of recording a fallback.
    base_client: ChatClient = build_client(settings, max_retry_wait=30.0) if live else OfflineRanker()
    judge_mode = args.judge if args.judge != "auto" else ("llm" if live else "heuristic")
    judge = None
    if judge_mode == "llm":
        if not settings.llm_enabled:
            print("--judge llm needs an API key", file=sys.stderr)
            return 2
        judge = LLMJudge(build_client(settings, model=settings.resolved_judge_model, temperature=0.0,
                                      max_tokens=1500, max_retry_wait=30.0))

    variants = list(VARIANTS) if args.prompt == "all" else [args.prompt or settings.prompt_variant]
    for v in variants:
        if v not in VARIANTS:
            print(f"Unknown prompt variant {v!r}; choose from {', '.join(VARIANTS)} or 'all'", file=sys.stderr)
            return 2
    cases = load_cases(args.cases)
    if args.tags:
        wanted = {t.strip() for t in args.tags.split(",") if t.strip()}
        cases = [c for c in cases if wanted & set(c.get("tags", []))]
    judged = [c["id"] for c in cases if "judge" in c.get("tags", [])] if judge else []
    injection_only = args.inject == "llm_failures"
    latency_n = args.latency if args.latency is not None else (3 if live else 10)
    delay = args.delay if args.delay is not None else (1.0 if live else 0.0)

    reports = []
    for variant in variants:
        print(f"Variant {variant} ({'live ' + base_client.model if live else 'mock'})", flush=True)
        report = VariantReport(variant, prompt_chars=prompt_size(store, settings, variant))
        throttle = Throttle(report.telemetry, delay, enabled=live)
        report.injections = run_injections(store, settings, variant)
        if not injection_only:
            run_cases(cases, store, settings, variant, base_client, judge, throttle, report, live)
            latency_sample(store, settings, variant, base_client, throttle, report, latency_n)
        reports.append(report)

    header = {
        "Eval run": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "Git SHA": git_revision(),
        "Dataset": f"{Path(settings.data_path).name} ({len(store):,} rows, version {store.version})",
        "LLM model": (f"{settings.llm_provider} · {base_client.model}" if live else "offline mock (no LLM)"),
        "Judge": judge.name if judge else ("heuristic only" if live else "n/a (mock explanations)"),
        "Prompt": " vs ".join(variants),
        "Cases": f"{args.cases.name} ({len(cases)}{', tags ' + args.tags if args.tags else ''})",
    }
    text = render(reports, header, judged, injection_only)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    ok = all(r.hard_pass for r in reports)
    print(f"\nScorecard written to {out}")
    for r in reports:
        print(f"  {r.variant}: hard metrics {'PASS' if r.hard_pass else 'FAIL'}; "
              f"judge Acceptable {_pct(r.acceptable('judge'))}; heuristic Acceptable {_pct(r.acceptable('heuristic'))}")
    if len(reports) > 1:
        winner, _ = choose_winner(reports)
        print(f"  A/B winner: {winner}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
