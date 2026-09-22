import pandas as pd

from src.drift.detector import DriftResult
from src.evaluation.evaluator import ModelEvaluator
from src.models.pytorch_trainer import PyTorchModelTrainer
from src.pipeline import RecommendationPipeline


class NoDriftDetector:
    def detect(self, reference_data, current_data):
        return DriftResult(
            drift_detected=False,
            drift_score=0.0,
            method="test",
        )


def test_pipeline_without_drift():

    data = pd.DataFrame({
        "user_id": [1],
        "item_id": [10],
        "reward": [1],
        "timestamp": pd.to_datetime([
            "2026-01-01"
        ]),
    })

    trainer = PyTorchModelTrainer(
        embedding_dim=8,
        epochs=1,
        batch_size=1,
    )

    evaluator = ModelEvaluator(
        top_k=1,
    )

    pipeline = RecommendationPipeline(
        drift_detector=NoDriftDetector(),
        model_trainer=trainer,
        evaluator=evaluator,
    )

    result = pipeline.run(
        reference_data=data,
        current_data=data,
        evaluation_data=data,
        champion_model=None,
        champion_trainer=trainer,
    )

    assert result.retraining_triggered is False
    assert result.model_promoted is False
    assert result.evaluation_result is None