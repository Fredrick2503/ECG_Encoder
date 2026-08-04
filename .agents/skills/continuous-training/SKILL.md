---
name: continuous-training
description: Manage continuous model improvement by scheduling retraining, resuming interrupted training, running hyperparameter searches, and promoting better-performing models based on evaluation results.
risk: medium
source: project
---

# Continuous Training

## Objective

Continuously improve model performance throughout the project.

This skill is responsible for coordinating retraining, resuming interrupted
training, running hyperparameter optimization, and identifying candidate models
for promotion.

It does **not** implement models or manage experiment tracking.

---

# When to Use

Use this skill whenever:

- training is interrupted
- new data becomes available
- hyperparameter tuning is required
- model performance needs improvement
- scheduled retraining is requested
- a better model should be searched for

---

# Responsibilities

Manage:

- Training continuation
- Retraining schedules
- Hyperparameter optimization
- Training retries
- Best-model selection
- Training recommendations

---

# Workflow

For every training cycle:

1. Review the latest training results.
2. Determine whether retraining is needed.
3. Resume or schedule training if appropriate.
4. Compare new results with the current best model.
5. Recommend promotion of a better model if performance improves.

---

# Dependencies

Works closely with:

- MLflow Manager
- Model Engineering
- Evaluation & Validation

---

# Rules

- Never overwrite the current best model without evaluation.
- Resume interrupted training whenever possible.
- Compare new models against the current baseline.
- Record recommendations instead of making deployment decisions.
- Leave experiment tracking to the MLflow Manager.

---

# Success Criteria

The skill is successful when:

- Interrupted training is resumed when possible.
- Retraining is scheduled when appropriate.
- Hyperparameter searches complete successfully.
- Better-performing models are identified.
- Training recommendations are clearly reported.