---
name: mlflow-manager
description: Manage MLflow experiments, runs, artifacts, metrics, model registry, and checkpoints. Use whenever training starts, resumes, completes, or when experiment metadata needs to be recorded or queried.
risk: medium
source: project
---

# MLflow Manager

## Objective

Manage the complete MLflow lifecycle for the project.

This skill is responsible for ensuring every training run is tracked,
artifacts are organized, metrics are recorded, and models are properly
registered.

It does **not** train models or evaluate results.

---

# When to Use

Use this skill whenever:

- starting a training run
- resuming training
- completing training
- logging metrics
- saving artifacts
- registering a model
- loading a registered model
- comparing experiment runs

---

# Responsibilities

Manage:

- Experiments
- Runs
- Parameters
- Metrics
- Artifacts
- Checkpoints
- Model Registry
- Model Versions
- Run Metadata

---

# Workflow

For every experiment:

1. Create or identify the experiment.
2. Start or resume an MLflow run.
3. Log parameters.
4. Record metrics during training.
5. Save artifacts and checkpoints.
6. Register the model if appropriate.
7. Close the run cleanly.

---

# Artifacts to Manage

Track:

- Model checkpoints
- Training logs
- Validation metrics
- Configuration files
- Performance plots
- Exported models

---

# Rules

- Every training run must belong to an experiment.
- Log all important hyperparameters.
- Keep experiment names consistent.
- Never overwrite registered models.
- Use versioning for model updates.
- Preserve experiment history.

---

# Success Criteria

The skill is successful when:

- Every training run is tracked.
- Metrics are available for comparison.
- Artifacts are stored correctly.
- Models are versioned.
- Experiments are reproducible.