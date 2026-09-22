import pandas as pd
import torch

from src.evaluation.evaluator import ModelEvaluator
from src.models.pytorch_trainer import PyTorchModelTrainer
from src.models.two_tower import TwoTowerRecommender


def test_evaluator():

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