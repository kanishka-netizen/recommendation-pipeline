from pathlib import Path

from src.data.movielens import load_movielens_ratings
from src.models.pytorch_trainer import PyTorchModelTrainer


DATA_PATH = Path("data/movielens/ratings.dat")
MODEL_PATH = Path("artifacts/challenger.pt")


data = load_movielens_ratings(DATA_PATH)

print(f"Loaded {len(data)} interactions.")
print(f"Positive interactions: {(data['reward'] > 0).sum()}")

# Sort chronologically and keep only the oldest 80% for champion training.
# The newest 20% is reserved for challenger training and evaluation, so the
# champion must never be trained on those rows — training on held-out
# evaluation interactions would make the champion's metric artificially high
# and the champion/challenger comparison unfair.
data = data.sort_values("timestamp").reset_index(drop=True)
split_index = int(len(data) * 0.8)
training_data = data.iloc[:split_index].copy()

print(f"Training interactions (oldest 80%): {len(training_data)}")
print(
    f"Training window: "
    f"{training_data['timestamp'].min()} "
    f"to {training_data['timestamp'].max()}"
)

trainer = PyTorchModelTrainer(
    embedding_dim=64,
    epochs=5,
    batch_size=256,
    learning_rate=0.001,
)

model = trainer.train(training_data)

trainer.save(
    model,
    MODEL_PATH,
)

print(f"Model saved to: {MODEL_PATH}")