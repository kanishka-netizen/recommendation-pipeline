# recommendation-pipeline

An adaptive recommendation system that learns from user interactions, detects changes in user behaviour, and updates the recommendation model when needed.

The project is being developed as part of our academic project on an adaptive, drift-aware recommendation pipeline.

---

# what are we building?

The goal is to build a quiz platform that can eventually recommend quizzes to users based on their previous interactions and preferences.

The interesting part is that the recommendation model is not supposed to stay fixed forever.

As users interact with the platform, their behaviour can change. Our system is designed to:

1. Collect user interaction data
2. Generate recommendations
3. Continuously monitor incoming interactions
4. Detect data or performance drift
5. Retrain a challenger model using recent data
6. Evaluate the challenger against the existing model
7. Promote the challenger only when it shows significant improvement
8. Keep track of the model lifecycle through monitoring

In short:

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
Promote / Reject
