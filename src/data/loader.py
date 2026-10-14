from pathlib import Path
import pandas as pd

from src.quiz.interaction_events import CSV_READ_DTYPES


def adapt_quiz_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize quiz interaction records for the recommendation pipeline
    by mapping question_id to item_id if item_id is not already present.
    """
    if "question_id" in df.columns and "item_id" not in df.columns:
        df = df.copy()
        df["item_id"] = df["question_id"]
    return df


def load_interactions(
    path: str | Path,
    standardize: bool = True,
) -> pd.DataFrame:
    """
    Load user interaction data from a CSV file.

    Parameters
    ----------
    path:
        Path to the interaction CSV.
    standardize:
        If True, standardizes question_id to item_id for pipeline compatibility.

    Returns
    -------
    pd.DataFrame
        Loaded interaction data.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Interaction data not found: {path}")

    df = pd.read_csv(path, dtype=CSV_READ_DTYPES)

    if standardize:
        df = adapt_quiz_interactions(df)

    return df