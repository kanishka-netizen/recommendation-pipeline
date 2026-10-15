"""
Regression tests for Stage 3 - data leakage prevention.

These tests prove that the champion-training split and the pipeline''s
challenger-training step cannot access interactions reserved for evaluation.

Three things are verified:
1. The 80/20 chronological split used in train_model.py and run_pipeline.py
   produces strictly non-overlapping training and evaluation windows.
2. get_recent_interactions, when applied to the training slice, cannot return
   any timestamp that falls inside the evaluation window.
3. When the pipeline runs, the data passed to trainer.train() contains no
   timestamps from evaluation_data (regression guard on pipeline.run itself).
"""

import pandas as pd
import pytest

from src.data.movielens import get_recent_interactions
from src.drift.detector import DriftResult
from src.evaluation.evaluator import EvaluationResult
from src.pipeline import RecommendationPipeline


# ---------------------------------------------------------------------------
# Helpers shared by multiple tests
# ---------------------------------------------------------------------------

def _make_chronological_dataset(n_rows=100):
    """
    Synthetic interaction dataset with n_rows rows, one per day,
    sorted chronologically. Mimics the shape that load_movielens_ratings
    produces after the sort-by-timestamp step.
    """
    return pd.DataFrame({
        "user_id": [i % 10 + 1 for i in range(n_rows)],
        "item_id": [i % 20 + 1 for i in range(n_rows)],
        "reward": [1] * n_rows,
        "timestamp": pd.date_range("2026-01-01", periods=n_rows, freq="D"),
    })


def _make_80_20_split(data):
    """
    Reproduce the exact same split logic used in train_model.py and
    run_pipeline.py so the tests are coupled to the real boundary.
    """
    data = data.sort_values("timestamp").reset_index(drop=True)
    split_index = int(len(data) * 0.8)
    training_data = data.iloc[:split_index].copy()
    held_out_data = data.iloc[split_index:].copy()
    return training_data, held_out_data


# ---------------------------------------------------------------------------
# Test 1 - the 80/20 split creates strictly non-overlapping timestamp sets
# ---------------------------------------------------------------------------

def test_training_split_excludes_all_evaluation_timestamps():
    """
    After the chronological 80/20 split, no timestamp present in the
    evaluation portion should appear in the training portion.
    """
    data = _make_chronological_dataset(n_rows=100)
    training_data, evaluation_data = _make_80_20_split(data)

    training_timestamps = set(training_data["timestamp"].tolist())
    evaluation_timestamps = set(evaluation_data["timestamp"].tolist())

    overlap = training_timestamps & evaluation_timestamps
    assert overlap == set(), (
        f"Leakage detected: {len(overlap)} timestamp(s) appear in both "
        f"training_data and evaluation_data after the 80/20 split."
    )


def test_training_window_strictly_precedes_evaluation_window():
    """
    The maximum timestamp in the training portion must be strictly less than
    the minimum timestamp in the evaluation portion.
    """
    data = _make_chronological_dataset(n_rows=100)
    training_data, evaluation_data = _make_80_20_split(data)

    assert training_data["timestamp"].max() < evaluation_data["timestamp"].min(), (
        "Training window overlaps with evaluation window: the champion could "
        "have been trained on evaluation-period interactions."
    )


# ---------------------------------------------------------------------------
# Test 2 - get_recent_interactions on the training slice stays within
#           the training window and cannot reach evaluation-period rows
# ---------------------------------------------------------------------------

def test_recent_interactions_from_training_slice_stays_within_training_window():
    """
    When get_recent_interactions is applied to training_data, every row it
    returns must have a timestamp <= training_data max timestamp.
    """
    data = _make_chronological_dataset(n_rows=100)
    training_data, evaluation_data = _make_80_20_split(data)

    recent = get_recent_interactions(training_data, days=30)

    assert not recent.empty, "get_recent_interactions returned nothing from training_data"

    training_max_ts = training_data["timestamp"].max()
    eval_min_ts = evaluation_data["timestamp"].min()

    assert recent["timestamp"].max() <= training_max_ts

    # No returned row must touch the evaluation window.
    leaked = recent[recent["timestamp"] >= eval_min_ts]
    assert leaked.empty, (
        f"get_recent_interactions returned {len(leaked)} row(s) whose timestamps "
        f"fall inside the evaluation window (>= {eval_min_ts})."
    )


# ---------------------------------------------------------------------------
# Test 3 - pipeline.run() does not pass evaluation_data timestamps to trainer
# ---------------------------------------------------------------------------

class _RecordingTrainer:
    """Mock trainer that records exactly what data it was trained on."""

    def __init__(self):
        self.trained_timestamps = None
        self.train_called = False

    def train(self, data):
        self.train_called = True
        self.trained_timestamps = set(data["timestamp"].tolist())
        return "mock-challenger-model"

    def save(self, model, path):
        pass


class _AlwaysDriftDetector:
    def detect(self, reference_data, current_data):
        return DriftResult(drift_detected=True, drift_score=0.99, method="test")


class _AlwaysPromoteEvaluator:
    def evaluate(self, champion, challenger, evaluation_data,
                 champion_trainer, challenger_trainer):
        return EvaluationResult(
            champion_metric=0.5,
            challenger_metric=0.8,
            improvement=0.3,
            statistically_significant=True,
        )


def test_pipeline_challenger_training_never_sees_evaluation_timestamps():
    """
    The data passed to trainer.train() during pipeline.run() must not contain
    any timestamp that appears in evaluation_data.
    """
    data = _make_chronological_dataset(n_rows=100)
    data = data.sort_values("timestamp").reset_index(drop=True)

    outer_split = int(len(data) * 0.8)
    reference_data = data.iloc[:outer_split].copy()
    current_data = data.iloc[outer_split:].copy()

    inner_split = int(len(current_data) * 0.8)
    training_current_data = current_data.iloc[:inner_split].copy()
    evaluation_data = current_data.iloc[inner_split:].copy()

    evaluation_timestamps = set(evaluation_data["timestamp"].tolist())

    recording_trainer = _RecordingTrainer()

    pipeline = RecommendationPipeline(
        drift_detector=_AlwaysDriftDetector(),
        model_trainer=recording_trainer,
        evaluator=_AlwaysPromoteEvaluator(),
        recent_days=9999,   # large window so all of training_current_data is used
    )

    pipeline.run(
        reference_data=reference_data,
        current_data=training_current_data,
        evaluation_data=evaluation_data,
        champion_model="mock-champion",
        champion_trainer=recording_trainer,
    )

    assert recording_trainer.train_called, "Trainer was never called"

    leaked = recording_trainer.trained_timestamps & evaluation_timestamps
    assert leaked == set(), (
        f"Pipeline passed {len(leaked)} evaluation-period timestamp(s) to "
        f"trainer.train()."
    )


def test_pipeline_no_leakage_when_evaluation_window_is_separate():
    """
    The max timestamp seen by trainer.train() must be strictly less than the
    min timestamp in evaluation_data.
    """
    data = _make_chronological_dataset(n_rows=100)
    data = data.sort_values("timestamp").reset_index(drop=True)

    outer_split = int(len(data) * 0.8)
    reference_data = data.iloc[:outer_split].copy()
    current_data = data.iloc[outer_split:].copy()

    inner_split = int(len(current_data) * 0.8)
    training_current_data = current_data.iloc[:inner_split].copy()
    evaluation_data = current_data.iloc[inner_split:].copy()

    eval_min_ts = evaluation_data["timestamp"].min()

    recording_trainer = _RecordingTrainer()

    pipeline = RecommendationPipeline(
        drift_detector=_AlwaysDriftDetector(),
        model_trainer=recording_trainer,
        evaluator=_AlwaysPromoteEvaluator(),
        recent_days=9999,
    )

    pipeline.run(
        reference_data=reference_data,
        current_data=training_current_data,
        evaluation_data=evaluation_data,
        champion_model="mock-champion",
        champion_trainer=recording_trainer,
    )

    assert recording_trainer.train_called

    if recording_trainer.trained_timestamps:
        max_training_ts = max(recording_trainer.trained_timestamps)
        assert max_training_ts < eval_min_ts, (
            f"trainer.train() saw timestamp {max_training_ts} which is >= "
            f"the evaluation window start {eval_min_ts}."
        )
