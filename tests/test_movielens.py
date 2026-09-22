import pandas as pd

from pathlib import Path

from src.data.movielens import (
    load_movielens_ratings,
    get_recent_interactions,
)


def test_load_movielens():

    path = Path("data/movielens/ratings.dat")

    data = load_movielens_ratings(path)

    assert not data.empty

    assert "user_id" in data.columns
    assert "item_id" in data.columns
    assert "rating" in data.columns
    assert "reward" in data.columns
    assert "timestamp" in data.columns


def test_reward_conversion():

    path = Path("data/movielens/ratings.dat")

    data = load_movielens_ratings(path)

    assert set(data["reward"].unique()).issubset({0, 1})

def test_recent_interactions():

    path = Path(
        "data/movielens/ratings.dat"
    )

    data = load_movielens_ratings(path)

    recent = get_recent_interactions(
        data,
        days=30,
    )

    assert not recent.empty

    assert (
        recent["timestamp"].min()
        >= data["timestamp"].max()
        - pd.Timedelta(days=30)
    )