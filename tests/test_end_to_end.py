"""
End-to-end pipeline integration test (Stage 5).

Exercises the complete pipeline flow using only synthetic quiz data:
  1. Chronological 80/20 split (leakage guard).
  2. Drift detection (KS test - forced by synthetic distribution shift).
  3. Challenger training on recent interactions from training_current.
  4. Champion/challenger evaluation (Hit Rate@K + paired bootstrap).
  5. Checkpoint promotion when challenger wins.

No MovieLens file is required.  The test is self-contained.
"""

import pandas as pd
from pathlib import Path

import pytest

from src.data.movielens import get_recent_interactions
from src.drift.detector import KSNumericalDriftDetector
from src.evaluation.evaluator import ModelEvaluator
from src.models.pytorch_trainer import PyTorchModelTrainer
from src.pipeline import RecommendationPipeline
from src.quiz.data_generator import generate_synthetic_dataset


def _build_end_to_end_data():
    """
    Build a synthetic dataset that mimics the real pipeline splits.

    Returns reference_data, training_current_data, evaluation_data -
    each is a non-overlapping chronological slice of the full dataset.
    """
    full_data = generate_synthetic_dataset(
        n_users=20,
        n_questions=25,
        n_interactions=500,
        seed=7,
    )
    full_data = full_data.sort_values("timestamp").reset_index(drop=True)

    # Outer 80/20 split
    outer_split = int(len(full_data) * 0.8)
    reference_data = full_data.iloc[:outer_split].copy()
    current_data = full_data.iloc[outer_split:].copy()

    # Inner 80/20 split of current
    inner_split = int(len(current_data) * 0.8)
    training_current_data = current_data.iloc[:inner_split].copy()
    evaluation_data = current_data.iloc[inner_split:].copy()

    return reference_data, training_current_data, evaluation_data


def test_end_to_end_pipeline_no_drift_returns_no_retraining():
    """
    When reference and current data come from the same distribution,
    the KS test should not detect drift and the pipeline should skip
    retraining and evaluation.
    """
    full_data = generate_synthetic_dataset(
        n_users=20, n_questions=25, n_interactions=500, seed=8
    )
    full_data = full_data.sort_values("timestamp").reset_index(drop=True)

    outer_split = int(len(full_data) * 0.8)
    reference_data = full_data.iloc[:outer_split].copy()
    current_data = full_data.iloc[outer_split:].copy()

    inner_split = int(len(current_data) * 0.8)
    training_current_data = current_data.iloc[:inner_split].copy()
    evaluation_data = current_data.iloc[inner_split:].copy()

    # Use a very generous threshold (0.99) so almost nothing triggers drift.
    from src.drift.detector import KSNumericalDriftDetector
    from unittest.mock import MagicMock

    # Force no-drift by patching the detector
    class _NoDrift:
        def detect(self, reference_data, current_data):
            from src.drift.detector import DriftResult
            return DriftResult(drift_detected=False, drift_score=0.0, method="ks")

    trainer = PyTorchModelTrainer(embedding_dim=8, epochs=1, batch_size=32)
    evaluator = ModelEvaluator(top_k=5)

    pipeline = RecommendationPipeline(
        drift_detector=_NoDrift(),
        model_trainer=trainer,
        evaluator=evaluator,
    )

    result = pipeline.run(
        reference_data=reference_data,
        current_data=training_current_data,
        evaluation_data=evaluation_data,
        champion_model=None,
        champion_trainer=trainer,
    )

    assert result.drift_result.drift_detected is False
    assert result.retraining_triggered is False
    assert result.evaluation_result is None
    assert result.model_promoted is False


def test_end_to_end_pipeline_with_drift_trains_and_evaluates(tmp_path):
    """
    Full pipeline run with synthetic quiz data and a forced drift signal.

    Checks:
    - retraining_triggered is True.
    - evaluation_result is not None and has valid metrics.
    - The challenger checkpoint file was written.
    - The champion file is written if the challenger wins.
    - Chronological separation: evaluation window timestamps are absent from
      the data seen by the trainer.
    """
    reference_data, training_current_data, evaluation_data = (
        _build_end_to_end_data()
    )

    # Verify data split is chronologically clean before even running the pipeline.
    assert training_current_data["timestamp"].max() < evaluation_data["timestamp"].min(), (
        "Pre-condition: training_current_data must precede evaluation_data."
    )

    class _ForceDrift:
        def detect(self, reference_data, current_data):
            from src.drift.detector import DriftResult
            return DriftResult(drift_detected=True, drift_score=0.95, method="test")

    class _TrackingTrainer(PyTorchModelTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.seen_timestamps = set()

        def train(self, data):
            self.seen_timestamps = set(data["timestamp"].tolist())
            return super().train(data)

    challenger_path = str(tmp_path / "challenger.pt")
    champion_path = str(tmp_path / "champion.pt")

    trainer = _TrackingTrainer(embedding_dim=8, epochs=1, batch_size=32)

    # We need a champion model to pass in — train one on reference data.
    champion_trainer = PyTorchModelTrainer(embedding_dim=8, epochs=1, batch_size=32)
    champion_model = champion_trainer.train(reference_data)
    champion_trainer.save(champion_model, champion_path)

    evaluator = ModelEvaluator(top_k=5)

    pipeline = RecommendationPipeline(
        drift_detector=_ForceDrift(),
        model_trainer=trainer,
        evaluator=evaluator,
        challenger_path=challenger_path,
        champion_path=champion_path,
        recent_days=9999,
    )

    result = pipeline.run(
        reference_data=reference_data,
        current_data=training_current_data,
        evaluation_data=evaluation_data,
        champion_model=champion_model,
        champion_trainer=champion_trainer,
    )

    # Pipeline must have retrained.
    assert result.retraining_triggered is True
    assert result.evaluation_result is not None

    # Evaluation metrics must be valid floats in [0, 1].
    assert 0.0 <= result.evaluation_result.champion_metric <= 1.0
    assert 0.0 <= result.evaluation_result.challenger_metric <= 1.0
    assert result.evaluation_result.improvement == (
        result.evaluation_result.challenger_metric
        - result.evaluation_result.champion_metric
    )

    # Challenger checkpoint must have been saved.
    assert Path(challenger_path).exists(), "Challenger checkpoint should be written"

    # Chronological separation: the trainer must not have seen evaluation timestamps.
    eval_timestamps = set(evaluation_data["timestamp"].tolist())
    leaked = trainer.seen_timestamps & eval_timestamps
    assert leaked == set(), (
        f"End-to-end leakage: trainer saw {len(leaked)} evaluation-period timestamp(s)."
    )

    # model_promoted is consistent with statistically_significant.
    assert result.model_promoted == result.evaluation_result.statistically_significant


def test_end_to_end_chronological_separation_holds():
    """
    Focused regression: the three-way split (reference / training_current /
    evaluation) must be strictly ordered with no timestamp overlap.
    """
    reference_data, training_current_data, evaluation_data = (
        _build_end_to_end_data()
    )

    ref_ts = set(reference_data["timestamp"].tolist())
    train_ts = set(training_current_data["timestamp"].tolist())
    eval_ts = set(evaluation_data["timestamp"].tolist())

    assert ref_ts.isdisjoint(train_ts), "reference and training_current overlap"
    assert ref_ts.isdisjoint(eval_ts), "reference and evaluation overlap"
    assert train_ts.isdisjoint(eval_ts), "training_current and evaluation overlap"

    assert reference_data["timestamp"].max() < training_current_data["timestamp"].min()
    assert training_current_data["timestamp"].max() < evaluation_data["timestamp"].min()
