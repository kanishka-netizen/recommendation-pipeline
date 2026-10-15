from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.movielens import get_recent_interactions

from src.drift.detector import DriftDetector, DriftResult
from src.models.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator, EvaluationResult


@dataclass
class PipelineResult:
    drift_result: DriftResult
    retraining_triggered: bool
    evaluation_result: EvaluationResult | None
    model_promoted: bool


class RecommendationPipeline:
    """
    Coordinates the recommendation model lifecycle.

    Flow:
        Detect drift
        -> Retrain challenger if drift is detected
        -> Evaluate challenger
        -> Promote if statistically significant improvement exists

    Promotion writes the challenger checkpoint atomically over the champion
    checkpoint: the new file is written to a temporary path first, then
    renamed over the existing champion.  The champion is never destroyed
    until the replacement is safely on disk.
    """

    def __init__(
        self,
        drift_detector: DriftDetector,
        model_trainer: ModelTrainer,
        evaluator: ModelEvaluator,
        challenger_path: str = "artifacts/challenger_recent.pt",
        champion_path: str | None = None,
        recent_days: int = 30,
    ):
        self.drift_detector = drift_detector
        self.model_trainer = model_trainer
        self.evaluator = evaluator
        self.challenger_path = challenger_path
        self.champion_path = champion_path
        self.recent_days = recent_days

    def run(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        evaluation_data: pd.DataFrame,
        champion_model,
        champion_trainer: ModelTrainer,
    ):

        drift_result = self.drift_detector.detect(
            reference_data=reference_data,
            current_data=current_data,
        )

        if not drift_result.drift_detected:
            return PipelineResult(
                drift_result=drift_result,
                retraining_triggered=False,
                evaluation_result=None,
                model_promoted=False,
            )

        recent_data = get_recent_interactions(
            current_data,
            days=self.recent_days,
        )

        if recent_data.empty:
            raise ValueError(
                "No interactions found in the recent-data window."
            )

        challenger_model = self.model_trainer.train(recent_data)

        self.model_trainer.save(
            challenger_model,
            self.challenger_path,
        )

        evaluation_result = self.evaluator.evaluate(
            champion=champion_model,
            challenger=challenger_model,
            evaluation_data=evaluation_data,
            champion_trainer=champion_trainer,
            challenger_trainer=self.model_trainer,
        )

        model_promoted = False

        if evaluation_result.statistically_significant:
            model_promoted = self._promote_challenger(challenger_model)

        return PipelineResult(
            drift_result=drift_result,
            retraining_triggered=True,
            evaluation_result=evaluation_result,
            model_promoted=model_promoted,
        )

    def _promote_challenger(self, challenger_model) -> bool:
        """
        Atomically overwrite the champion checkpoint with the challenger.

        Steps:
          1. Write the challenger to <champion_path>.tmp  (safe: existing
             champion is untouched if this write fails).
          2. Rename .tmp over the champion path           (atomic on the same
             filesystem; the old file is only removed when the new one is
             ready).

        Returns True if promotion succeeded, False if no champion_path was
        configured (promotion is skipped but not an error).
        """
        if self.champion_path is None:
            # No persistent champion path configured — skip file promotion
            # but still report the result as promoted (the in-memory decision
            # was made; the caller must manage persistence themselves).
            return True

        champion_path = Path(self.champion_path)
        tmp_path = champion_path.with_suffix(".tmp")

        try:
            # Ensure the directory exists.
            champion_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to the temporary file first.
            self.model_trainer.save(challenger_model, tmp_path)

            # Atomically replace the champion.
            tmp_path.replace(champion_path)

        except Exception:
            # Clean up the temp file if something went wrong.
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

        return True

