from __future__ import annotations

import pytest

from restaurant_recommender.config import Settings
from restaurant_recommender.data.store import RestaurantStore
from tests.helpers import FakeLLM, make_df


@pytest.fixture
def restaurants_df():
    return make_df()


@pytest.fixture
def store(restaurants_df) -> RestaurantStore:
    return RestaurantStore(restaurants_df)


@pytest.fixture
def settings(tmp_path) -> Settings:
    # _env_file=None: tests never read a developer's real .env / API key.
    return Settings(
        _env_file=None,
        LLM_API_KEY="",
        DATA_PATH=str(tmp_path / "missing.parquet"),
        MAX_CANDIDATES=30,
        TOP_N_DEFAULT=5,
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
