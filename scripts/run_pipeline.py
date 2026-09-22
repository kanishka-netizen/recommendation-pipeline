from pathlib import Path

from src.data.movielens import load_movielens_ratings
from src.models.pytorch_trainer import PyTorchModelTrainer
from src.drift.detector import KSNumericalDriftDetector
from src.evaluation.evaluator import ModelEvaluator
from src.pipeline import RecommendationPipeline


DATA_PATH = Path("data/movielens/ratings.dat")
CHAMPION_PATH = Path("artifacts/challenger.pt")
CHALLENGER_PATH = Path("artifacts/challenger_recent.pt")


data = load_movielens_ratings(DATA_PATH)

print(f"Loaded {len(data)} interactions.")

# Sort interactions chronologically
data = data.sort_values("timestamp").reset_index(drop=True)

# Split older interactions from newer interactions
split_index = int(len(data) * 0.8)

reference_data = data.iloc[:split_index].copy()
current_data = data.iloc[split_index:].copy()

# Temporary evaluation set for the end-to-end pipeline demo
# Keep the most recent interactions as a held-out evaluation set.
evaluation_split = int(len(current_data) * 0.8)

training_current_data = current_data.iloc[:evaluation_split].copy()
evaluation_data = current_data.iloc[evaluation_split:].copy()

print(f"Reference interactions: {len(reference_data)}")
print(f"Current interactions: {len(current_data)}")
print(f"Evaluation interactions: {len(evaluation_data)}")

trainer = PyTorchModelTrainer(
    embedding_dim=64,
    epochs=5,
    batch_size=256,
    learning_rate=0.001,
)

champion_model = trainer.load(CHAMPION_PATH)


drift_detector = KSNumericalDriftDetector()
evaluator = ModelEvaluator(top_k=10)

pipeline = RecommendationPipeline(
    drift_detector=drift_detector,
    model_trainer=trainer,
    evaluator=evaluator,
    challenger_path=str(CHALLENGER_PATH),
    recent_days=30,
)
result = pipeline.run(
    reference_data=reference_data,
    current_data=training_current_data,
    evaluation_data=evaluation_data,
    champion_model=champion_model,
    champion_trainer=trainer,
)

print("\n=== Pipeline Result ===")
print(f"Drift detected: {result.drift_result.drift_detected}")
print(f"Drift score: {result.drift_result.drift_score:.4f}")
print(f"Retraining triggered: {result.retraining_triggered}")

if result.evaluation_result is not None:
    print(
        f"Champion metric: "
        f"{result.evaluation_result.champion_metric:.4f}"
    )
    print(
        f"Challenger metric: "
        f"{result.evaluation_result.challenger_metric:.4f}"
    )
    print(
        f"Improvement: "
        f"{result.evaluation_result.improvement:.4f}"
    )
    print(
        f"Model promoted: "
        f"{result.model_promoted}"
    )