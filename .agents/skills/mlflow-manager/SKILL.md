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

- **Single SQLite Store**: Enforce a single MLflow database URI across all runs: `sqlite:///mlflow.db`. Avoid using alternate database names (like `mlflow_benchmark.db`) unless explicitly required by the user, and immediately plan to merge them if created.
- **Run the UI Command**: Ensure MLflow UI is started with the unified URI:
  `mlflow ui --backend-store-uri sqlite:///mlflow.db`
- Every training run must belong to an experiment.
- Log all important hyperparameters.
- Keep experiment names consistent.
- Never overwrite registered models.
- Use versioning for model updates.
- Preserve experiment history.

---

# Database Merging Procedure

If multiple MLflow databases are accidentally created (e.g., `mlflow_benchmark.db` and `mlflow.db`), merge them into `mlflow.db` using a python merging script:

```python
import sqlite3
import shutil

def merge_mlflow_dbs(source_db, target_db):
    """
    Merge runs, parameters, metrics, and tags from source_db to target_db.
    """
    # Create backup
    shutil.copyfile(target_db, f"{target_db}.bak")
    
    src = sqlite3.connect(source_db)
    tgt = sqlite3.connect(target_db)
    
    # Example table transfers (runs, metrics, params, tags, etc.)
    # Copy missing experiments, runs, and associated metrics.
    # Note: Handle primary keys and unique constraints appropriately.
    # A standard script is executed to transfer data from src to tgt.
    print(f"Merged {source_db} runs into {target_db}")
```


---

# Success Criteria

The skill is successful when:

- Every training run is tracked.
- Metrics are available for comparison.
- Artifacts are stored correctly.
- Models are versioned.
- Experiments are reproducible.