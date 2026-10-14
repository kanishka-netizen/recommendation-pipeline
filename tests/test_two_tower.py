import pandas as pd
import torch

from src.models.two_tower import TwoTowerRecommender


def test_model_forward():

    model = TwoTowerRecommender(
        num_users=10,
        num_items=20,
        embedding_dim=64,
    )

    user_ids = torch.tensor([0, 1, 2])
    item_ids = torch.tensor([3, 4, 5])

    scores = model(
        user_ids,
        item_ids
    )

    assert scores.shape == (3,)


def test_user_embedding():

    model = TwoTowerRecommender(
        num_users=10,
        num_items=20,
        embedding_dim=64,
    )

    user_ids = torch.tensor([0, 1])

    embeddings = model.encode_users(
        user_ids
    )

    assert embeddings.shape == (2, 64)


def test_item_embedding():

    model = TwoTowerRecommender(
        num_users=10,
        num_items=20,
        embedding_dim=64,
    )

    item_ids = torch.tensor([0, 1, 2])

    embeddings = model.encode_items(
        item_ids
    )

    assert embeddings.shape == (3, 64)
def test_recommendations():

    from src.models.pytorch_trainer import PyTorchModelTrainer

    trainer = PyTorchModelTrainer()

    trainer.user_to_index = {
        1: 0,
    }

    trainer.item_to_index = {
        10: 0,
        20: 1,
        30: 2,
    }

    trainer.user_positive_items = {
        0: {0},
    }

    model = TwoTowerRecommender(
        num_users=1,
        num_items=3,
        embedding_dim=8,
    )

    recommendations = trainer.recommend(
        model,
        user_id=1,
        top_k=2,
    )

    recommended_items = [
        item_id
        for item_id, score in recommendations
    ]

    assert len(recommendations) == 2
    assert 10 not in recommended_items


def test_train_on_quiz_data_and_generate_recommendations():
    """
    Train PyTorchModelTrainer on synthetic quiz interactions containing
    question_id, user_id, reward, and timestamp, and verify recommendations.
    """
    from src.models.pytorch_trainer import PyTorchModelTrainer
    from src.quiz.data_generator import generate_synthetic_dataset

    quiz_data = generate_synthetic_dataset(
        n_users=10,
        n_questions=15,
        n_interactions=120,
        seed=42,
    )

    assert "question_id" in quiz_data.columns
    assert "item_id" not in quiz_data.columns

    trainer = PyTorchModelTrainer(
        embedding_dim=16,
        epochs=2,
        batch_size=32,
        learning_rate=0.01,
    )

    model = trainer.train(quiz_data)
    assert model is not None

    # Pick an active user in the training set
    target_user = list(trainer.user_to_index.keys())[0]
    recommendations = trainer.recommend(
        model,
        user_id=target_user,
        top_k=5,
    )

    assert len(recommendations) > 0
    assert len(recommendations) <= 5

    target_user_idx = trainer.user_to_index[target_user]
    consumed_indices = trainer.user_positive_items.get(target_user_idx, set())
    consumed_question_ids = {
        item_id for item_id, idx in trainer.item_to_index.items()
        if idx in consumed_indices
    }

    previous_score = float("inf")
    for qid, score in recommendations:
        # Original question_id format preserved
        assert qid in quiz_data["question_id"].values
        # Consumed questions are filtered out
        assert qid not in consumed_question_ids
        # Recommendations are sorted by score descending
        assert score <= previous_score
        previous_score = score


def test_quiz_model_persistence_and_original_question_ids(tmp_path):
    """
    Verify model checkpoint saving and loading maintains original question IDs
    and identical recommendation scores.
    """
    from src.models.pytorch_trainer import PyTorchModelTrainer
    from src.quiz.data_generator import generate_synthetic_dataset

    quiz_data = generate_synthetic_dataset(
        n_users=8,
        n_questions=12,
        n_interactions=80,
        seed=42,
    )

    trainer = PyTorchModelTrainer(
        embedding_dim=16,
        epochs=2,
        batch_size=32,
    )
    model = trainer.train(quiz_data)

    target_user = list(trainer.user_to_index.keys())[0]
    original_recs = trainer.recommend(model, user_id=target_user, top_k=5)

    checkpoint_file = tmp_path / "quiz_model.pt"
    trainer.save(model, checkpoint_file)
    assert checkpoint_file.exists()

    new_trainer = PyTorchModelTrainer(embedding_dim=16)
    loaded_model = new_trainer.load(checkpoint_file)

    assert new_trainer.user_to_index == trainer.user_to_index
    assert new_trainer.item_to_index == trainer.item_to_index
    assert new_trainer.user_positive_items == trainer.user_positive_items

    loaded_recs = new_trainer.recommend(loaded_model, user_id=target_user, top_k=5)

    assert len(loaded_recs) == len(original_recs)
    for (orig_qid, orig_score), (load_qid, load_score) in zip(original_recs, loaded_recs):
        assert orig_qid == load_qid
        assert abs(orig_score - load_score) < 1e-5


def test_negative_sampling_bounded_failure_when_user_interacted_with_all_items():
    """
    When a user has interacted with every available item, negative sampling
    must fail explicitly with a bounded failure condition rather than hanging indefinitely.
    """
    import pytest
    import pandas as pd
    from src.models.pytorch_trainer import PyTorchModelTrainer

    # User 1 interacted with all 2 items; user 2 interacted with 1 item
    data = pd.DataFrame({
        "user_id": [1, 1, 2],
        "question_id": [101, 102, 101],
        "reward": [1, 1, 1],
        "timestamp": [100.0, 200.0, 300.0],
    })

    trainer = PyTorchModelTrainer(epochs=1, batch_size=2)

    with pytest.raises(ValueError, match="interacted with all"):
        trainer.train(data)


def test_negative_sampling_requires_at_least_two_items():
    """
    Datasets with only 1 item cannot sample negatives and must raise an explicit ValueError.
    """
    import pytest
    import pandas as pd
    from src.models.pytorch_trainer import PyTorchModelTrainer

    data = pd.DataFrame({
        "user_id": [1, 2],
        "item_id": [101, 101],
        "reward": [1, 1],
        "timestamp": [100.0, 200.0],
    })

    trainer = PyTorchModelTrainer(epochs=1)

    with pytest.raises(ValueError, match="at least 2 distinct items"):
        trainer.train(data)


def test_load_interactions_quiz_adapter(tmp_path):
    """
    Test load_interactions adapter standardizes question_id to item_id.
    """
    from src.data.loader import load_interactions
    from src.quiz.data_generator import write_sample_dataset

    csv_path = tmp_path / "sample_quiz.csv"
    write_sample_dataset(path=csv_path, n_users=5, n_questions=10, n_interactions=30)

    loaded_df = load_interactions(csv_path)

    assert "question_id" in loaded_df.columns
    assert "item_id" in loaded_df.columns
    assert (loaded_df["item_id"] == loaded_df["question_id"]).all()
    # selected_answer should remain string type
    assert pd.api.types.is_string_dtype(loaded_df["selected_answer"])


def test_get_recent_interactions_numeric_and_datetime():
    """
    Verify get_recent_interactions works with both float Unix timestamps and datetime timestamps.
    """
    import pandas as pd
    from src.data.movielens import get_recent_interactions

    # 1. Numeric timestamp (seconds)
    # 3 days span: 0s, 86400s (1 day), 172800s (2 days)
    df_numeric = pd.DataFrame({
        "item_id": [1, 2, 3],
        "timestamp": [0.0, 86400.0, 172800.0],
    })
    recent_num = get_recent_interactions(df_numeric, days=1)
    assert len(recent_num) == 2
    assert recent_num["item_id"].tolist() == [2, 3]

    # 2. Datetime timestamp
    df_datetime = pd.DataFrame({
        "item_id": [1, 2, 3],
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="D"),
    })
    recent_dt = get_recent_interactions(df_datetime, days=1)
    assert len(recent_dt) == 2
    assert recent_dt["item_id"].tolist() == [2, 3]