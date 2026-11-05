# syntax=docker/dockerfile:1
# One image, two roles:  `api` (FastAPI, default)  and  `ui` (Streamlit).
#   docker build -t restaurant-recommender:1.0.0 .
#   docker run --env-file .env -p 8000:8000 restaurant-recommender:1.0.0 api
#   docker run -e RECOMMENDER_API_URL=http://host.docker.internal:8000 -p 8501:8501 restaurant-recommender:1.0.0 ui
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    DATA_PATH=/app/data/processed/restaurants.parquet \
    HF_HOME=/tmp/hf-cache

WORKDIR /app

# Dependencies first (cached layer), pinned to the versions the test suite passed with.
COPY requirements.lock ./
RUN pip install -r requirements.lock

# Application code and runtime config.
COPY src ./src
COPY ui ./ui
COPY .streamlit ./.streamlit
COPY tests/eval ./tests/eval
COPY data ./data
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

# Bake the cleaned dataset into the image so containers start without network access
# to Hugging Face. Skipped when data/processed/restaurants.parquet is already in the
# build context, or with --build-arg INGEST_AT_BUILD=0 (then it runs on first start).
ARG INGEST_AT_BUILD=1
RUN if [ "$INGEST_AT_BUILD" = "1" ] && [ ! -f "$DATA_PATH" ]; then \
        python -m restaurant_recommender.data.ingest --out "$DATA_PATH"; \
    fi

RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8000 8501
ENTRYPOINT ["entrypoint.sh"]
CMD ["api"]
