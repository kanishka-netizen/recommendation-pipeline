from pathlib import Path

import pandas as pd


def load_movielens_ratings(path: str | Path) -> pd.DataFrame:
    """
    Load MovieLens 1M ratings and convert them into
    the interaction format used by the recommender.

    MovieLens format:
        UserID::MovieID::Rating::Timestamp

    Our format:
        user_id
        item_id
        reward
        timestamp

    Ratings >= 4 are treated as positive interactions.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"MovieLens ratings file not found: {path}"
        )

    data = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=[
            "user_id",
            "item_id",
            "rating",
            "timestamp",
    ])

    # Convert the timestamp from Unix seconds
    # into a pandas datetime.
    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        unit="s",
        utc=True,
    )

    # Convert ratings into a binary interaction signal.
    data["reward"] = (
        data["rating"] >= 4
    ).astype(int)

    return data[
        [
            "user_id",
            "item_id",
            "rating",
            "reward",
            "timestamp",
        ]
    ]
def get_recent_interactions(data, days=30):
    """
    Return interactions from the most recent time window.

    Parameters
    ----------
    data:
        Interaction DataFrame containing a timestamp column.

    days:
        Number of recent days to keep.
    """

    if data.empty:
        return data.copy()

    if "timestamp" not in data.columns:
        raise ValueError(
            "Interaction data must contain a timestamp column."
        )

    latest_timestamp = data["timestamp"].max()

    cutoff = latest_timestamp - pd.Timedelta(
        days=days,
    )

    recent_data = data[
        data["timestamp"] >= cutoff
    ].copy()

    return recent_data