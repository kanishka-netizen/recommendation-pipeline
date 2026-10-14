from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class EvaluationResult:
    champion_metric: float
    challenger_metric: float
    improvement: float
    statistically_significant: bool


class ModelEvaluator:
    """
    Evaluates and compares recommendation models using paired Hit Rate@K.
    """

    DEFAULT_MIN_USERS: int = 1
    DEFAULT_MIN_BOOTSTRAP_SAMPLES: int = 5

    def __init__(
        self,
        top_k: int = 10,
        min_users: int = DEFAULT_MIN_USERS,
        min_bootstrap_samples: int = DEFAULT_MIN_BOOTSTRAP_SAMPLES,
    ):
        """
        Parameters
        ----------
        top_k:
            Cutoff rank K for Hit Rate@K recommendations.
        min_users:
            Minimum common eligible users required to perform an evaluation.
            If fewer common users exist, evaluate() returns zeroed metrics.
        min_bootstrap_samples:
            Minimum sample size required to perform paired bootstrap significance
            testing. It is consistently at least min_users to avoid claiming
            significance on tiny, statistically indefensible sample sizes.
        """
        self.top_k = top_k
        self.min_users = max(1, min_users)
        self.min_bootstrap_samples = max(min_bootstrap_samples, self.min_users)

    def _bootstrap_significance(
        self,
        champion_hits: list[int] | np.ndarray,
        challenger_hits: list[int] | np.ndarray,
        iterations: int = 1000,
        min_samples: int | None = None,
    ) -> bool:
        """
        Paired bootstrap hypothesis test for H0: E[challenger - champion] <= 0
        against H1: E[challenger - champion] > 0.

        Resamples per-user differences (challenger - champion) with replacement.
        Returns True if the lower 2.5% quantile of the bootstrap mean difference
        distribution is strictly positive.
        """
        if champion_hits is None or challenger_hits is None:
            return False

        if len(champion_hits) == 0 or len(challenger_hits) == 0:
            return False

        if len(champion_hits) != len(challenger_hits):
            return False

        threshold = self.min_bootstrap_samples if min_samples is None else min_samples
        if len(champion_hits) < threshold:
            return False

        champ_arr = np.asarray(champion_hits, dtype=float)
        chall_arr = np.asarray(challenger_hits, dtype=float)

        paired_differences = chall_arr - champ_arr
        sample_size = len(paired_differences)

        rng = np.random.default_rng(42)
        differences = []

        for _ in range(iterations):
            indices = rng.integers(0, sample_size, size=sample_size)
            differences.append(paired_differences[indices].mean())

        lower_bound = np.percentile(differences, 2.5)
        return bool(lower_bound > 0)

    def _user_hits(
        self,
        model: Any,
        trainer: Any,
        evaluation_data: pd.DataFrame,
    ) -> dict[int, int]:
        """
        Compute binary Hit Rate@K outcome per eligible user for a single model.

        Eligible users are those who have positive interactions in evaluation_data
        (reward > 0) AND exist in the trainer's user mapping.

        Returns
        -------
        dict[int, int]
            Mapping from user_id to binary outcome (1 for hit, 0 for miss).
        """
        if evaluation_data is None or evaluation_data.empty:
            return {}

        required_cols = {"user_id", "item_id", "reward"}
        if not required_cols.issubset(evaluation_data.columns):
            return {}

        positive_data = evaluation_data[evaluation_data["reward"] > 0]
        if positive_data.empty:
            return {}

        positive_items_by_user = (
            positive_data.groupby("user_id")["item_id"].apply(set).to_dict()
        )

        user_hits: dict[int, int] = {}
        for user_id in sorted(positive_items_by_user.keys()):
            if not hasattr(trainer, "user_to_index") or user_id not in trainer.user_to_index:
                continue

            actual_items = positive_items_by_user[user_id]

            try:
                recommendations = trainer.recommend(
                    model,
                    user_id=user_id,
                    top_k=self.top_k,
                )
            except Exception:
                continue

            recommended_items = {
                item_id for item_id, score in recommendations
            }

            hit = 1 if (actual_items & recommended_items) else 0
            user_hits[user_id] = hit

        return user_hits

    def _hit_rate(self, model: Any, trainer: Any, evaluation_data: pd.DataFrame) -> float:
        """
        Compute aggregate Hit Rate@K for a single model across its eligible users.
        """
        user_hits = self._user_hits(
            model,
            trainer,
            evaluation_data,
        )

        if not user_hits:
            return 0.0

        return float(np.mean(list(user_hits.values())))

    def evaluate(
        self,
        champion: Any,
        challenger: Any,
        evaluation_data: pd.DataFrame,
        champion_trainer: Any,
        challenger_trainer: Any,
    ) -> EvaluationResult:
        """
        Compare champion and challenger using paired Hit Rate@K.

        1. Evaluates each model independently to produce a mapping of user_id -> hit.
        2. Explicitly intersects the user IDs returned by both models.
        3. Aligns outcomes using the same ordered sequence of common user IDs.
        4. Computes champion and challenger Hit Rate@K on that shared set.
        5. Computes paired bootstrap significance on the user-level differences.
        """
        champ_user_hits = self._user_hits(
            champion,
            champion_trainer,
            evaluation_data,
        )

        chall_user_hits = self._user_hits(
            challenger,
            challenger_trainer,
            evaluation_data,
        )

        common_user_ids = sorted(
            set(champ_user_hits.keys()) & set(chall_user_hits.keys())
        )

        if len(common_user_ids) < self.min_users:
            return EvaluationResult(
                champion_metric=0.0,
                challenger_metric=0.0,
                improvement=0.0,
                statistically_significant=False,
            )

        champion_hits = [champ_user_hits[uid] for uid in common_user_ids]
        challenger_hits = [chall_user_hits[uid] for uid in common_user_ids]

        champion_metric = float(np.mean(champion_hits))
        challenger_metric = float(np.mean(challenger_hits))
        improvement = challenger_metric - champion_metric

        statistically_significant = self._bootstrap_significance(
            champion_hits,
            challenger_hits,
        )

        return EvaluationResult(
            champion_metric=champion_metric,
            challenger_metric=challenger_metric,
            improvement=improvement,
            statistically_significant=statistically_significant,
        )