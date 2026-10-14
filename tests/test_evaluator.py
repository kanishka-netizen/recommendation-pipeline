import pandas as pd
import torch

from src.evaluation.evaluator import ModelEvaluator
from src.models.pytorch_trainer import PyTorchModelTrainer
from src.models.two_tower import TwoTowerRecommender


def test_evaluator():
    """
    1. Existing basic evaluator smoke test with two-tower model and trainer.
    """
    # Small evaluation dataset.
    evaluation_data = pd.DataFrame({
        "user_id": [1, 2],
        "item_id": [10, 20],
        "reward": [1, 1],
        "timestamp": pd.to_datetime([
            "2026-01-01",
            "2026-01-01",
        ]),
    })

    # Create a tiny model.
    model = TwoTowerRecommender(
        num_users=2,
        num_items=2,
        embedding_dim=8,
    )

    trainer = PyTorchModelTrainer()

    trainer.user_to_index = {
        1: 0,
        2: 1,
    }

    trainer.item_to_index = {
        10: 0,
        20: 1,
    }

    trainer.user_positive_items = {
        0: set(),
        1: set(),
    }

    evaluator = ModelEvaluator(
        top_k=1,
    )

    result = evaluator.evaluate(
        champion=model,
        challenger=model,
        evaluation_data=evaluation_data,
        champion_trainer=trainer,
        challenger_trainer=trainer,
    )

    assert 0.0 <= result.champion_metric <= 1.0
    assert 0.0 <= result.challenger_metric <= 1.0
    assert result.improvement == (
        result.challenger_metric
        - result.champion_metric
    )


class DummyTrainer:
    """Mock trainer that maps user_id to pre-determined recommendations or exceptions."""

    def __init__(self, user_to_index, recommendations=None, failing_users=None):
        self.user_to_index = user_to_index
        self.recommendations = recommendations or {}
        self.failing_users = set(failing_users or [])

    def recommend(self, model, user_id, top_k=10):
        if user_id in self.failing_users:
            raise RuntimeError(f"Simulated failure for user {user_id}")
        return self.recommendations.get(user_id, [])[:top_k]


def test_different_user_mappings_only_common_eligible_contribute():
    """
    2. Champion and challenger with different user mappings: only their common
    eligible users contribute to Hit Rate@K metrics and improvement.
    """
    evaluation_data = pd.DataFrame({
        "user_id": [101, 102, 103, 104, 105],
        "item_id": [1, 2, 3, 4, 5],
        "reward": [1, 1, 1, 1, 1],
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="D"),
    })

    # Champion knows {101, 102, 103}
    champion_trainer = DummyTrainer(
        user_to_index={101: 0, 102: 1, 103: 2},
        recommendations={
            101: [(1, 0.9)],   # hit on item 1 (must be excluded; not in challenger)
            102: [(99, 0.9)],  # miss on item 2
            103: [(3, 0.9)],   # hit on item 3
        },
    )

    # Challenger knows {102, 103, 104}
    challenger_trainer = DummyTrainer(
        user_to_index={102: 0, 103: 1, 104: 2},
        recommendations={
            102: [(2, 0.9)],   # hit on item 2
            103: [(3, 0.9)],   # hit on item 3
            104: [(99, 0.9)],  # miss on item 4 (must be excluded; not in champion)
        },
    )

    evaluator = ModelEvaluator(top_k=1)
    result = evaluator.evaluate(
        champion="champ_model",
        challenger="chall_model",
        evaluation_data=evaluation_data,
        champion_trainer=champion_trainer,
        challenger_trainer=challenger_trainer,
    )

    # Common users: {102, 103} (size 2)
    # Champion: user 102 -> 0, user 103 -> 1 => 1/2 = 0.5
    # Challenger: user 102 -> 1, user 103 -> 1 => 2/2 = 1.0
    assert result.champion_metric == 0.5
    assert result.challenger_metric == 1.0
    assert result.improvement == 0.5


def test_per_user_pairing_when_outcomes_differ():
    """
    3. Correct per-user pairing when outcomes differ.
    Verifies that _user_hits returns user-keyed mappings and evaluate aligns
    outcomes per user ID rather than treating outcomes as independent bags.
    """
    evaluation_data = pd.DataFrame({
        "user_id": [1, 2],
        "item_id": [10, 20],
        "reward": [1, 1],
        "timestamp": pd.date_range("2026-01-01", periods=2, freq="D"),
    })

    # User 1: champ hits (10), chall misses (99) -> diff = -1
    # User 2: champ misses (99), chall hits (20) -> diff = +1
    champ_trainer = DummyTrainer(
        user_to_index={1: 0, 2: 1},
        recommendations={1: [(10, 0.9)], 2: [(99, 0.9)]},
    )
    chall_trainer = DummyTrainer(
        user_to_index={1: 0, 2: 1},
        recommendations={1: [(99, 0.9)], 2: [(20, 0.9)]},
    )

    evaluator = ModelEvaluator(top_k=1)

    # Verify per-user mapping returned by _user_hits
    champ_hits = evaluator._user_hits("champ", champ_trainer, evaluation_data)
    chall_hits = evaluator._user_hits("chall", chall_trainer, evaluation_data)
    assert champ_hits == {1: 1, 2: 0}
    assert chall_hits == {1: 0, 2: 1}

    result = evaluator.evaluate(
        champion="champ",
        challenger="chall",
        evaluation_data=evaluation_data,
        champion_trainer=champ_trainer,
        challenger_trainer=chall_trainer,
    )
    assert result.champion_metric == 0.5
    assert result.challenger_metric == 0.5
    assert result.improvement == 0.0
    assert result.statistically_significant is False


def test_trainer_unexpectedly_skipping_a_user():
    """
    4. A trainer unexpectedly skipping a user (e.g. exception/inference failure):
    both metrics and significance testing use only users actually returned by both models.
    """
    evaluation_data = pd.DataFrame({
        "user_id": [1, 2, 3],
        "item_id": [10, 20, 30],
        "reward": [1, 1, 1],
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="D"),
    })

    champ_trainer = DummyTrainer(
        user_to_index={1: 0, 2: 1, 3: 2},
        recommendations={
            1: [(99, 0.9)],  # miss (0)
            2: [(20, 0.9)],  # hit (1) - but challenger will fail on user 2!
            3: [(30, 0.9)],  # hit (1)
        },
    )
    # Challenger fails on user 2
    chall_trainer = DummyTrainer(
        user_to_index={1: 0, 2: 1, 3: 2},
        recommendations={
            1: [(10, 0.9)],  # hit (1)
            3: [(30, 0.9)],  # hit (1)
        },
        failing_users=[2],
    )

    evaluator = ModelEvaluator(top_k=1)
    result = evaluator.evaluate(
        champion="champ",
        challenger="chall",
        evaluation_data=evaluation_data,
        champion_trainer=champ_trainer,
        challenger_trainer=chall_trainer,
    )

    # User 2 must be excluded from both models because challenger skipped user 2.
    # Common evaluated users: {1, 3} (size 2).
    # Champion over {1, 3}: user 1 (0), user 3 (1) => 1/2 = 0.5 (NOT 2/3).
    # Challenger over {1, 3}: user 1 (1), user 3 (1) => 2/2 = 1.0.
    assert result.champion_metric == 0.5
    assert result.challenger_metric == 1.0
    assert result.improvement == 0.5


def test_empty_evaluation_data_or_no_positive_rewards():
    """
    5. Empty evaluation data or evaluation data with zero positive rewards
    must return zeroed metrics and not raise errors.
    """
    user_map = {1: 0, 2: 1}
    trainer = DummyTrainer(user_to_index=user_map, recommendations={1: [(10, 0.9)]})
    evaluator = ModelEvaluator(top_k=1)

    # 5a. Empty dataframe
    empty_df = pd.DataFrame()
    res_empty = evaluator.evaluate("c", "t", empty_df, trainer, trainer)
    assert res_empty.champion_metric == 0.0
    assert res_empty.challenger_metric == 0.0
    assert res_empty.improvement == 0.0
    assert res_empty.statistically_significant is False
    assert evaluator._user_hits("c", trainer, empty_df) == {}

    # 5b. Non-empty dataframe, but all reward == 0
    zero_reward_df = pd.DataFrame({
        "user_id": [1, 2],
        "item_id": [10, 20],
        "reward": [0, 0],
        "timestamp": pd.date_range("2026-01-01", periods=2, freq="D"),
    })
    res_zero = evaluator.evaluate("c", "t", zero_reward_df, trainer, trainer)
    assert res_zero.champion_metric == 0.0
    assert res_zero.challenger_metric == 0.0
    assert res_zero.improvement == 0.0
    assert res_zero.statistically_significant is False
    assert evaluator._user_hits("c", trainer, zero_reward_df) == {}


def test_no_common_eligible_users():
    """
    6. When champion and challenger have non-overlapping user mappings,
    no common eligible users exist.
    """
    evaluation_data = pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "item_id": [10, 20, 30, 40],
        "reward": [1, 1, 1, 1],
        "timestamp": pd.date_range("2026-01-01", periods=4, freq="D"),
    })

    champ_trainer = DummyTrainer(user_to_index={1: 0, 2: 1}, recommendations={})
    chall_trainer = DummyTrainer(user_to_index={3: 0, 4: 1}, recommendations={})

    evaluator = ModelEvaluator(top_k=1)
    result = evaluator.evaluate(
        champion="champ",
        challenger="chall",
        evaluation_data=evaluation_data,
        champion_trainer=champ_trainer,
        challenger_trainer=chall_trainer,
    )

    assert result.champion_metric == 0.0
    assert result.challenger_metric == 0.0
    assert result.improvement == 0.0
    assert result.statistically_significant is False


def test_min_users_threshold_not_met():
    """
    7. When min_users threshold is not satisfied by common eligible users,
    evaluate() returns zeroed metrics without crashing.
    """
    evaluation_data = pd.DataFrame({
        "user_id": [1, 2],
        "item_id": [10, 20],
        "reward": [1, 1],
        "timestamp": pd.date_range("2026-01-01", periods=2, freq="D"),
    })

    user_map = {1: 0, 2: 1}
    trainer = DummyTrainer(
        user_to_index=user_map,
        recommendations={1: [(10, 0.9)], 2: [(20, 0.9)]},
    )

    # Require at least 5 eligible users, but only 2 common users exist
    evaluator = ModelEvaluator(top_k=1, min_users=5)
    result = evaluator.evaluate(
        champion="champ",
        challenger="chall",
        evaluation_data=evaluation_data,
        champion_trainer=trainer,
        challenger_trainer=trainer,
    )

    assert result.champion_metric == 0.0
    assert result.challenger_metric == 0.0
    assert result.improvement == 0.0
    assert result.statistically_significant is False


def test_bootstrap_significance_empty_unequal_insufficient_samples():
    """
    8. Empty bootstrap inputs, unequal input lengths, and insufficient sample sizes.
    Statistical assumption: Paired bootstrap testing requires non-empty, equal-length
    paired vectors with sample size >= min_bootstrap_samples to avoid degenerate
    variance and false statistical claims.
    """
    evaluator = ModelEvaluator(min_bootstrap_samples=5)

    # Empty inputs
    assert evaluator._bootstrap_significance([], []) is False
    assert evaluator._bootstrap_significance([1], []) is False
    assert evaluator._bootstrap_significance(None, [1]) is False

    # Unequal lengths
    assert evaluator._bootstrap_significance([1, 0], [1]) is False
    assert evaluator._bootstrap_significance([1, 1, 1], [1, 1]) is False

    # Insufficient sample sizes (N < min_bootstrap_samples)
    assert evaluator._bootstrap_significance([0], [1]) is False  # N=1
    assert evaluator._bootstrap_significance([0, 0], [1, 1]) is False  # N=2 < 5
    assert evaluator._bootstrap_significance([0, 0, 0, 0], [1, 1, 1, 1]) is False  # N=4 < 5


def test_challenger_wins_consistently_significance():
    """
    9. Challenger wins consistently: positive improvement and statistical significance.
    Statistical assumption: When N=25 >= min_bootstrap_samples users all have a positive
    paired difference (challenger hit = 1, champion hit = 0), every bootstrap mean
    difference is +1.0, placing the lower 2.5% quantile at +1.0 (> 0).
    """
    user_ids = list(range(1, 26))  # 25 users >= min_bootstrap_samples (5)
    eval_rows = []
    champ_recs = {}
    chall_recs = {}

    for uid in user_ids:
        eval_rows.append({
            "user_id": uid,
            "item_id": uid * 10,
            "reward": 1,
            "timestamp": pd.Timestamp("2026-01-01"),
        })
        champ_recs[uid] = [(9999, 0.1)]  # champion misses
        chall_recs[uid] = [(uid * 10, 0.9)]  # challenger hits

    eval_df = pd.DataFrame(eval_rows)
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    champ_trainer = DummyTrainer(user_to_index=user_map, recommendations=champ_recs)
    chall_trainer = DummyTrainer(user_to_index=user_map, recommendations=chall_recs)

    evaluator = ModelEvaluator(top_k=1)
    result = evaluator.evaluate(
        champion="champ",
        challenger="chall",
        evaluation_data=eval_df,
        champion_trainer=champ_trainer,
        challenger_trainer=chall_trainer,
    )

    assert result.champion_metric == 0.0
    assert result.challenger_metric == 1.0
    assert result.improvement == 1.0
    assert result.statistically_significant is True


def test_identical_model_outcomes_zero_improvement_no_significance():
    """
    10. Identical model outcomes: zero improvement and no statistical significance.
    Statistical assumption: When both models produce identical recommendations for all users
    across N=25 users, every paired difference is 0. The bootstrap distribution of mean
    differences is degenerate at 0.0, so the 2.5th percentile is 0.0 (not > 0).
    """
    user_ids = list(range(1, 26))
    eval_rows = []
    recs = {}

    for uid in user_ids:
        eval_rows.append({
            "user_id": uid,
            "item_id": uid * 10,
            "reward": 1,
            "timestamp": pd.Timestamp("2026-01-01"),
        })
        # Both models recommend the exact target item
        recs[uid] = [(uid * 10, 0.9)]

    eval_df = pd.DataFrame(eval_rows)
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    trainer = DummyTrainer(user_to_index=user_map, recommendations=recs)

    evaluator = ModelEvaluator(top_k=1)
    result = evaluator.evaluate(
        champion="champ",
        challenger="chall",
        evaluation_data=eval_df,
        champion_trainer=trainer,
        challenger_trainer=trainer,
    )

    assert result.champion_metric == 1.0
    assert result.challenger_metric == 1.0
    assert result.improvement == 0.0
    assert result.statistically_significant is False