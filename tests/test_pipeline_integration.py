import pandas as pd

from src.pipeline import RecommendationPipeline
from src.drift.detector import KSNumericalDriftDetector
from src.evaluation.evaluator import EvaluationResult


class MockTrainer:
    def __init__(self):
        self.train_called = False

    def train(self, data):
        self.train_called = True
        return "challenger-model"

    def save(self, model, path):
        pass
class MockEvaluator:
    def evaluate(
        self,
        champion,
        challenger,
        evaluation_data,
        champion_trainer,
        challenger_trainer,
    ):
        return EvaluationResult(
            champion_metric=0.70,
            challenger_metric=0.75,
            improvement=0.05,
            statistically_significant=True,
        )


def test_real_drift_detector_triggers_pipeline():

    reference_data = pd.DataFrame({
        "age": [20,21,22,23,24,25,26,27,28,29],
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="D"),
    })

    current_data = pd.DataFrame({
        "age": [80,81,82,83,84,85,86,87,88,89],
        "timestamp": pd.date_range("2026-02-01", periods=10, freq="D"),
    })

    evaluation_data = current_data.copy()

    trainer = MockTrainer()

    pipeline = RecommendationPipeline(
        drift_detector=KSNumericalDriftDetector(),
        model_trainer=trainer,
        evaluator=MockEvaluator(),
    )

    result = pipeline.run(
        reference_data=reference_data,
        current_data=current_data,
        evaluation_data=evaluation_data,
        champion_model="champion-model",
        champion_trainer=trainer,
    )

    assert result.drift_result.drift_detected is True
    assert result.retraining_triggered is True
    assert trainer.train_called is True
    assert result.evaluation_result is not None
    assert result.model_promoted is True
def test_pipeline_drift_retraining_and_evaluation():

    reference_data = pd.DataFrame({
        "age": [20,21,22,23,24,25,26,27,28,29],
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="D"),
    })

    current_data = pd.DataFrame({
        "age": [80,81,82,83,84,85,86,87,88,89],
        "timestamp": pd.date_range("2026-02-01", periods=10, freq="D"),
    })

    evaluation_data = current_data.copy()

    trainer = MockTrainer()

    pipeline = RecommendationPipeline(
        drift_detector=KSNumericalDriftDetector(),
        model_trainer=trainer,
        evaluator=MockEvaluator(),
    )

    result = pipeline.run(
        reference_data=reference_data,
        current_data=current_data,
        evaluation_data=evaluation_data,
        champion_model="champion-model",
        champion_trainer=trainer,
    )

    assert result.drift_result.drift_detected is True
    assert result.retraining_triggered is True
    assert result.evaluation_result is not None
def test_pipeline_uses_recent_data():
    reference_data = pd.DataFrame({
        "age": [20,21,22,23,24,25,26,27,28,29],
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="D"),
    })

    current_data = pd.DataFrame({
        "age": [80, 81, 82, 83],
        "user_id": [1, 1, 2, 2],
        "item_id": [101, 102, 103, 104],
        "reward": [1, 1, 1, 1],
        "timestamp": pd.to_datetime([
            "2026-01-01",
            "2026-01-05",
            "2026-02-05",
            "2026-02-10",
        ]),
})

    evaluation_data = current_data.copy()

    class TrackingTrainer(MockTrainer):
        def train(self, data):
            self.train_called = True
            self.training_data = data.copy()
            return "challenger-model"

    trainer = TrackingTrainer()

    pipeline = RecommendationPipeline(
        drift_detector=KSNumericalDriftDetector(),
        model_trainer=trainer,
        evaluator=MockEvaluator(),
        recent_days=30,
    )

    pipeline.run(
        reference_data=reference_data,
        current_data=current_data,
        evaluation_data=evaluation_data,
        champion_model="champion-model",
        champion_trainer=trainer,
    )

    assert trainer.train_called is True
    assert len(trainer.training_data) == 2
    assert trainer.training_data["item_id"].tolist() == [103, 104]