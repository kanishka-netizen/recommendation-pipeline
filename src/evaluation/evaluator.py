from dataclasses import dataclass

import pandas as pd

import numpy as np

@dataclass
class EvaluationResult:
    champion_metric: float
    challenger_metric: float
    improvement: float
    statistically_significant: bool


class ModelEvaluator:
    """
    Evaluates and compares recommendation models.
    """
    def _bootstrap_significance(
        self,
        champion_hits,
        challenger_hits,
        iterations=1000,
    ):
        if not champion_hits or not challenger_hits:
            return False

        rng = np.random.default_rng(42)

        champion_hits = np.array(champion_hits)
        challenger_hits = np.array(challenger_hits)

        sample_size = min(
            len(champion_hits),
            len(challenger_hits),
        )

        differences = []

        for _ in range(iterations):
            indices = rng.integers(
                0,
                sample_size,
                size=sample_size,
            )

            champion_sample = champion_hits[
                indices
            ]

            challenger_sample = challenger_hits[
                indices
            ]

            difference = (
                challenger_sample.mean()
                - champion_sample.mean()
            )

            differences.append(difference)

        lower_bound = np.percentile(
            differences,
            2.5,
        )

        return lower_bound > 0
    def __init__(self, top_k: int = 10):
        self.top_k = top_k

    def _user_hits(self, model, trainer, evaluation_data):
        positive_data = evaluation_data[evaluation_data["reward"] > 0]

        if positive_data.empty:
            return []

        user_hits = []

        for user_id, group in positive_data.groupby("user_id"):
            if user_id not in trainer.user_to_index:
                continue

            actual_items = set(group["item_id"].tolist())

            recommendations = trainer.recommend(
                model,
                user_id=user_id,
                top_k=self.top_k,
            )

            recommended_items = {
                item_id for item_id, score in recommendations
            }

            hit = 1 if actual_items & recommended_items else 0
            user_hits.append(hit)

        return user_hits


    def _hit_rate(self, model, trainer, evaluation_data):
        user_hits = self._user_hits(
            model,
            trainer,
            evaluation_data,
        )

        if not user_hits:
            return 0.0

        return sum(user_hits) / len(user_hits)

    def evaluate(
        self,
        champion,
        challenger,
        evaluation_data,
        champion_trainer,
        challenger_trainer,
    ) -> EvaluationResult:
        """
        Compare champion and challenger using Hit Rate@K.
        """

        champion_hits = self._user_hits(
            champion,
            champion_trainer,
            evaluation_data,
        )

        challenger_hits = self._user_hits(
            challenger,
            challenger_trainer,
            evaluation_data,
        )

        champion_metric = (
            sum(champion_hits) / len(champion_hits)
            if champion_hits
            else 0.0
        )

        challenger_metric = (
            sum(challenger_hits) / len(challenger_hits)
            if challenger_hits
            else 0.0
        )

        improvement = challenger_metric - champion_metric

        statistically_significant = self._bootstrap_significance(
            champion_hits,
            challenger_hits,
        )

        return EvaluationResult(
            champion_metric=champion_metric,
            challenger_metric=challenger_metric,
            improvement=improvement,
            statistically_significant=(
                statistically_significant
            ),
        )