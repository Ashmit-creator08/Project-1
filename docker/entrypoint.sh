#!/bin/sh
# Container entrypoint. Roles: api (default) | ui | ingest | eval | <any command>.
set -e

DATA_PATH="${DATA_PATH:-/app/data/processed/restaurants.parquet}"

ensure_data() {
    if [ ! -f "$DATA_PATH" ]; then
        echo "Dataset not found at $DATA_PATH; running ingest (first start only)..."
        python -m restaurant_recommender.data.ingest --out "$DATA_PATH"
    fi
}

role="${1:-api}"
[ "$#" -gt 0 ] && shift

case "$role" in
    api)
        ensure_data
        # One worker by default: the response cache, rate limiter and /metrics are
        # in-process. FastAPI runs the sync handler in a thread pool, so one worker
        # still serves concurrent requests.
        exec uvicorn restaurant_recommender.api.main:app \
            --host 0.0.0.0 \
            --port "${PORT:-8000}" \
            --workers "${WEB_CONCURRENCY:-1}" \
            --proxy-headers \
            --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-127.0.0.1}" \
            --no-server-header \
            "$@"
        ;;
    ui)
        # In API mode (RECOMMENDER_API_URL set) the UI never touches the dataset.
        [ -z "$RECOMMENDER_API_URL" ] && ensure_data
        exec streamlit run ui/app.py \
            --server.address 0.0.0.0 \
            --server.port "${PORT:-8501}" \
            "$@"
        ;;
    ingest)
        exec python -m restaurant_recommender.data.ingest "$@"
        ;;
    eval)
        ensure_data
        exec python -m restaurant_recommender.eval.run "$@"
        ;;
    *)
        exec "$role" "$@"
        ;;
esac
