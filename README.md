# Recommendation Pipeline

An adaptive recommendation system that learns from user interactions, detects changes in user behaviour, and updates the recommendation model when needed.

This project is being developed as part of an academic project on an adaptive, drift-aware recommendation pipeline. It aims to recommend quizzes based on users' previous interactions and preferences while monitoring behavioural changes and updating the model when appropriate.

## What are we building?

The system is designed to:

1. Collect user interaction data.
2. Generate recommendations.
3. Continuously monitor incoming interactions.
4. Detect data or performance drift.
5. Retrain a challenger model using recent data.
6. Evaluate the challenger against the existing model.
7. Promote the challenger only when it shows significant improvement.
8. Track the model lifecycle through monitoring.

```text
User Interactions
       ↓
  Recommendation
       ↓
Continuous Monitoring
       ↓
  Drift Detection
       ↓
Challenger Training
       ↓
    Evaluation
       ↓
  Promote/Reject
```

## Project layout

```text
scripts/
  train_model.py          Trains the initial model.
  run_pipeline.py         Runs the pipeline end-to-end.
  retrain_recent.py       Retrains on recent data.
  test_recommendations.py Checks saved-model recommendations.

src/
  pipeline.py             Recommendation pipeline orchestrator.
  data/                   Interaction data loading.
  drift/                  Drift detection.
  evaluation/             Model evaluation.
  models/                 Recommendation models and trainers.
  quiz/                   Quiz delivery, scoring, and interaction events.

tests/                    Automated test suite.
data/movielens/           MovieLens ratings data.
artifacts/                Saved model checkpoints.
docs/                     Schema and module documentation.
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Download MovieLens 1M and place `ratings.dat` at `data/movielens/ratings.dat`.

### 3. Train the initial model

```bash
python -m scripts.train_model
```

The training script should use the oldest 80% of interactions in chronological order to prevent data leakage into the evaluation window.

### 4. Run the pipeline

```bash
python -m scripts.run_pipeline
```

The intended workflow is to split historical and recent interactions chronologically, detect drift, train a challenger when appropriate, evaluate it on held-out data, and promote it only when the configured evaluation criteria are satisfied.

### 5. Run the tests

```bash
python -m pytest tests/ -v
```

## Data leakage safeguards

Chronological splitting helps prevent future interactions from leaking into training data used to evaluate model performance.

The leakage regression tests should verify that training and evaluation windows remain separate.

## Model promotion

When atomic promotion is enabled, the new checkpoint should be written to a temporary file before replacing the champion checkpoint. This helps preserve the existing model if writing the replacement fails.

## Quiz data support

The pipeline is intended to support quiz interactions alongside recommendation data. See `docs/interaction_schema.md` and `docs/quiz_module.md` for the relevant schemas and module documentation.

## Configuration

Review `scripts/run_pipeline.py` for the configured data paths, model checkpoint paths, recent-interaction window, and recommendation cutoff.