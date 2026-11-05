"""FastAPI app.

Endpoints: `POST /api/v1/recommendations`, `GET /api/v1/options`, `GET /health`, `GET /metrics`.

Run locally::

    uvicorn restaurant_recommender.api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import secrets
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from restaurant_recommender import __version__
from restaurant_recommender.api.ratelimit import TokenBucketLimiter
from restaurant_recommender.logging_config import configure_logging
from restaurant_recommender.config import Settings, get_settings
from restaurant_recommender.data.budget import band_range_label
from restaurant_recommender.data.store import DataUnavailableError, RestaurantStore
from restaurant_recommender.llm.client import ChatClient, build_client
from restaurant_recommender.models.schemas import Budget, PreferenceRequest, RecommendationResponse
from restaurant_recommender.services.recommend import RecommendationService
from restaurant_recommender.services.telemetry import Telemetry

log = logging.getLogger(__name__)

MAX_BODY_BYTES = 16 * 1024  # edge case OR-10
_UNSET: Any = object()


class UTF8JSONResponse(JSONResponse):
    # Explicit charset so clients such as Windows PowerShell 5.1 decode "₹" correctly.
    media_type = "application/json; charset=utf-8"


def create_app(
    *,
    settings: Settings | None = None,
    store: RestaurantStore | None = None,
    llm_client: ChatClient | None = _UNSET,
) -> FastAPI:
    """Build the app. Tests inject a store and a fake LLM client; production loads both from config."""
    settings = settings or get_settings()
    telemetry = Telemetry()
    limiter = TokenBucketLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level, settings.log_format)  # uvicorn only configures its own loggers
        app.state.store_error = None
        loaded = store
        if loaded is None:
            try:
                loaded = RestaurantStore.from_parquet(settings.data_path)
            except DataUnavailableError as exc:  # DI-13 / DI-14: serve 503, report unhealthy
                app.state.store_error = str(exc)
                log.error("%s", exc)
        client = build_client(settings) if llm_client is _UNSET else llm_client
        if client is None:
            log.warning("No LLM configured: recommendations will be rating-ranked with template text")
        else:
            log.info("LLM: provider=%s model=%s prompt=%s", settings.llm_provider, client.model, settings.prompt_variant)
        app.state.store = loaded
        app.state.llm = client
        app.state.service = (
            RecommendationService(loaded, client, settings, telemetry=telemetry) if loaded else None
        )
        yield

    app = FastAPI(
        title="Restaurant Recommender",
        version=__version__,
        description="Filters the Zomato dataset by your preferences, then asks an LLM (Groq) to rank and explain.",
        lifespan=lifespan,
        default_response_class=UTF8JSONResponse,
    )
    app.state.telemetry = telemetry
    app.state.limiter = limiter

    if settings.cors_origins:  # only needed when a browser app calls the API directly
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
            expose_headers=["X-Request-ID", "Retry-After"],
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Return type/loc/msg only. FastAPI's default echoes the raw input, which
        # fails to serialize for NaN (500 instead of 422) and can mirror huge strings.
        errors = [{k: err[k] for k in ("type", "loc", "msg") if k in err} for err in exc.errors()]
        return UTF8JSONResponse(status_code=422, content={"detail": errors})

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
            return UTF8JSONResponse(status_code=413, content={"detail": "Request body too large"})
        return await call_next(request)

    def rate_limit(request: Request) -> None:
        """Per-client token bucket (CF-08). Keyed by the direct client address."""
        client_key = request.client.host if request.client else "unknown"
        allowed, retry_after = limiter.check(client_key)
        if not allowed:
            telemetry.record_rate_limited()
            raise HTTPException(
                status_code=429,
                detail=f"Too many requests; limit is {limiter.per_minute} per minute",
                headers={"Retry-After": str(retry_after)},
            )

    def service_or_503(request: Request) -> RecommendationService:
        service = request.app.state.service
        if service is None:  # OR-05
            raise HTTPException(
                status_code=503,
                detail=request.app.state.store_error or "Restaurant data is not loaded",
            )
        return service

    @app.get("/health", tags=["ops"])
    def health(request: Request):
        state = request.app.state
        if state.store is None:  # OR-09
            return UTF8JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "detail": state.store_error or "data not loaded"},
            )
        service: RecommendationService = state.service
        return {
            "status": "ok",
            "restaurants": len(state.store),
            "data_version": state.store.version,
            "llm_enabled": state.llm is not None,
            "llm_provider": settings.llm_provider if state.llm is not None else None,
            "llm_model": state.llm.model if state.llm is not None else None,
            "prompt_variant": service.prompt.name,
            "cache_ttl_seconds": settings.cache_ttl_seconds,
            "rate_limit_per_minute": settings.rate_limit_per_minute,
            "version": __version__,
        }

    @app.get("/metrics", tags=["ops"])
    def metrics(request: Request, authorization: str | None = Header(default=None)):
        """Counters, fallback/cache rates, token usage and latency percentiles (no secrets).

        Protected by `METRICS_TOKEN` (Bearer) when that is set.
        """
        token = settings.metrics_token.get_secret_value()
        if token:
            supplied = (authorization or "").removeprefix("Bearer ").strip()
            if not secrets.compare_digest(supplied.encode(), token.encode()):
                raise HTTPException(status_code=401, detail="Missing or invalid metrics token",
                                    headers={"WWW-Authenticate": "Bearer"})
        snapshot = telemetry.snapshot()
        service = request.app.state.service
        snapshot["cache"]["entries"] = len(service.cache) if service and service.cache is not None else 0
        return snapshot

    @app.post(
        "/api/v1/recommendations",
        response_model=RecommendationResponse,
        tags=["recommendations"],
        dependencies=[Depends(rate_limit)],
        responses={
            422: {"description": "Invalid preferences"},
            429: {"description": "Rate limit exceeded"},
            503: {"description": "Data not loaded"},
        },
    )
    def recommend(
        prefs: PreferenceRequest,
        request: Request,
        response: Response,
        x_request_id: str | None = Header(default=None, max_length=64),
    ) -> RecommendationResponse:
        # Sync handler: FastAPI runs it in a worker thread, so the blocking LLM call
        # does not stall the event loop.
        service = service_or_503(request)
        request_id = x_request_id or uuid.uuid4().hex[:12]
        response.headers["X-Request-ID"] = request_id
        return service.recommend(prefs, request_id=request_id)

    @app.get("/api/v1/options", tags=["recommendations"])
    def options(request: Request, limit: int = 300):
        """Known locations and cuisines, most common first, for UI pickers."""
        service = service_or_503(request)
        limit = max(1, min(limit, 1000))
        return {
            "locations": [*service.store.cities(), *service.store.localities()][:limit],
            "cuisines": service.store.cuisines()[:limit],
            "budgets": {b.value: band_range_label(b) for b in Budget},
        }

    return app


app = create_app()
