from pathlib import Path

from src.models.pytorch_trainer import PyTorchModelTrainer


MODEL_PATH = Path("artifacts/challenger.pt")


trainer = PyTorchModelTrainer()

model = trainer.load(MODEL_PATH)

user_id = 1

recommendations = trainer.recommend(
    model,
    user_id=user_id,
    top_k=10,
)

print(f"\nTop 10 recommendations for user {user_id}:\n")

for item_id, score in recommendations:
    print(
        f"Item {item_id} - Score: {score:.4f}"
    )