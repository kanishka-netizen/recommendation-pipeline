from pathlib import Path

from src.data.movielens import (
    load_movielens_ratings,
    get_recent_interactions,
)
from src.models.pytorch_trainer import PyTorchModelTrainer


DATA_PATH = Path(
    "data/movielens/ratings.dat"
)

MODEL_PATH = Path(
    "artifacts/challenger_recent.pt"
)


# Load all available interaction data.
data = load_movielens_ratings(DATA_PATH)

print(f"Total interactions: {len(data)}")


# Select only the recent interaction window.
recent_data = get_recent_interactions(
    data,
    days=30,
)

print(
    f"Recent interactions: {len(recent_data)}"
)

print(
    f"Recent window: "
    f"{recent_data['timestamp'].min()} "
    f"to "
    f"{recent_data['timestamp'].max()}"
)


# Train a challenger using recent interactions.
trainer = PyTorchModelTrainer(
    embedding_dim=64,
    epochs=5,
    batch_size=256,
    learning_rate=0.001,
)

model = trainer.train(recent_data)


# Save the challenger.
trainer.save(
    model,
    MODEL_PATH,
)

print(
    f"Recent-data challenger saved to: "
    f"{MODEL_PATH}"
)