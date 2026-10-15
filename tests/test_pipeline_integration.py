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
# ---------------------------------------------------------------------------
# Stage 4 - Model promotion tests
# ---------------------------------------------------------------------------

from pathlib import Path

from src.drift.detector import DriftResult
from src.models.pytorch_trainer import PyTorchModelTrainer
from src.quiz.data_generator import generate_synthetic_dataset


class _PromotingMockTrainer:
    """
    Thin wrapper around a real PyTorchModelTrainer so promotion tests can
    verify that the champion file is actually updated on disk.
    """
    def __init__(self, real_trainer):
        self._real = real_trainer
        self.train_called = False
        self.trained_model = None

    @property
    def user_to_index(self):
        return self._real.user_to_index

    @property
    def item_to_index(self):
        return self._real.item_to_index

    @property
    def user_positive_items(self):
        return self._real.user_positive_items

    def train(self, data):
        self.train_called = True
        self.trained_model = self._real.train(data)
        return self.trained_model

    def save(self, model, path):
        self._real.save(model, path)

    def load(self, path):
        return self._real.load(path)

    def recommend(self, model, user_id, top_k=10):
        return self._real.recommend(model, user_id, top_k)


def _make_quiz_data():
    """Quiz data with user_id, question_id, reward, timestamp - ready for training."""
    return generate_synthetic_dataset(
        n_users=10, n_questions=15, n_interactions=120, seed=99
    )


class _AlwaysDrift:
    """Forces drift_detected=True so the pipeline always retrains."""
    def detect(self, reference_data, current_data):
        return DriftResult(drift_detected=True, drift_score=0.99, method="test")


class _WinsEvaluator:
    """Challenger always beats the champion (statistically significant)."""
    def evaluate(self, champion, challenger, evaluation_data,
                 champion_trainer, challenger_trainer):
        return EvaluationResult(
            champion_metric=0.4, challenger_metric=0.9,
            improvement=0.5, statistically_significant=True,
        )


class _LosesEvaluator:
    """Challenger never beats the champion."""
    def evaluate(self, champion, challenger, evaluation_data,
                 champion_trainer, challenger_trainer):
        return EvaluationResult(
            champion_metric=0.9, challenger_metric=0.4,
            improvement=-0.5, statistically_significant=False,
        )


def _setup_promotion_test(tmp_path, evaluator, quiz_data):
    """
    Common setup shared by all four promotion tests.

    Trains a champion on quiz_data, saves it to champion.pt in tmp_path,
    and returns a configured pipeline, the champion model, the champion file
    path, and the wrapped trainer.

    current_data in pipeline.run() is quiz_data (has user_id/question_id/
    reward/timestamp), so get_recent_interactions returns trainable rows.
    Drift is forced by _AlwaysDrift to guarantee retraining.
    """
    real_trainer = PyTorchModelTrainer(embedding_dim=8, epochs=1, batch_size=32)
    champion_model = real_trainer.train(quiz_data)

    champion_file = tmp_path / "champion.pt"
    real_trainer.save(champion_model, champion_file)

    wrapped = _PromotingMockTrainer(real_trainer)

    pipeline = RecommendationPipeline(
        drift_detector=_AlwaysDrift(),
        model_trainer=wrapped,
        evaluator=evaluator,
        challenger_path=str(tmp_path / "challenger.pt"),
        champion_path=str(champion_file),
        recent_days=9999,
    )

    return pipeline, champion_model, champion_file, wrapped


def test_qualifying_challenger_is_promoted_and_champion_file_updated(tmp_path):
    """
    When the challenger is statistically better, pipeline.run() must overwrite
    the champion checkpoint file.  After promotion the file must be loadable.
    """
    quiz_data = _make_quiz_data()
    pipeline, champion_model, champion_file, wrapped = (
        _setup_promotion_test(tmp_path, _WinsEvaluator(), quiz_data)
    )

    result = pipeline.run(
        reference_data=quiz_data.copy(),
        current_data=quiz_data,
        evaluation_data=quiz_data.copy(),
        champion_model=champion_model,
        champion_trainer=wrapped,
    )

    assert result.model_promoted is True
    assert champion_file.exists()

    verify_trainer = PyTorchModelTrainer(embedding_dim=8)
    loaded = verify_trainer.load(champion_file)
    assert loaded is not None


def test_rejected_challenger_does_not_replace_champion_file(tmp_path):
    """
    When the challenger does NOT win, the champion checkpoint must remain
    byte-for-byte identical to what it was before pipeline.run().
    """
    quiz_data = _make_quiz_data()
    pipeline, champion_model, champion_file, wrapped = (
        _setup_promotion_test(tmp_path, _LosesEvaluator(), quiz_data)
    )

    original_bytes = champion_file.read_bytes()

    result = pipeline.run(
        reference_data=quiz_data.copy(),
        current_data=quiz_data,
        evaluation_data=quiz_data.copy(),
        champion_model=champion_model,
        champion_trainer=wrapped,
    )

    assert result.model_promoted is False
    assert champion_file.read_bytes() == original_bytes


def test_promoted_champion_loads_correctly(tmp_path):
    """
    After promotion, a brand-new PyTorchModelTrainer can load the champion
    file and generate recommendations without error.
    """
    quiz_data = _make_quiz_data()
    pipeline, champion_model, champion_file, wrapped = (
        _setup_promotion_test(tmp_path, _WinsEvaluator(), quiz_data)
    )

    result = pipeline.run(
        reference_data=quiz_data.copy(),
        current_data=quiz_data,
        evaluation_data=quiz_data.copy(),
        champion_model=champion_model,
        champion_trainer=wrapped,
    )

    assert result.model_promoted is True

    fresh_trainer = PyTorchModelTrainer(embedding_dim=8)
    promoted_model = fresh_trainer.load(champion_file)

    sample_user = list(fresh_trainer.user_to_index.keys())[0]
    recs = fresh_trainer.recommend(promoted_model, user_id=sample_user, top_k=5)
    assert isinstance(recs, list) and len(recs) > 0


def test_promotion_status_flag_is_accurate(tmp_path):
    """
    model_promoted is True when the challenger wins and False when it loses.
    """
    quiz_data = _make_quiz_data()

    for evaluator_cls, expected in [(_WinsEvaluator, True), (_LosesEvaluator, False)]:
        pipeline, champion_model, champion_file, wrapped = (
            _setup_promotion_test(tmp_path, evaluator_cls(), quiz_data)
        )

        result = pipeline.run(
            reference_data=quiz_data.copy(),
            current_data=quiz_data,
            evaluation_data=quiz_data.copy(),
            champion_model=champion_model,
            champion_trainer=wrapped,
        )

        assert result.model_promoted is expected, (
            f"With {evaluator_cls.__name__}: expected model_promoted={expected}, "
            f"got {result.model_promoted}"
        )
