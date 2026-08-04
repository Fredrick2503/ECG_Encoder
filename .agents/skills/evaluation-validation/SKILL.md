---
name: evaluation-validation
description: Evaluate trained models, compare experimental results, validate performance, and generate evaluation reports. Use whenever a model has been trained or an experiment has completed.
risk: medium
source: project
---

# Evaluation & Validation

## Objective

Evaluate the performance of trained models and validate experimental results.

This skill is responsible for measuring model performance, comparing experiments,
generating evaluation artifacts, and determining whether a model satisfies the
project's acceptance criteria.

It does **not** train models, modify model architectures, or manage experiments.

---

# When to Use

Use this skill whenever:

- model training completes
- a new checkpoint is produced
- comparing multiple models
- validating an experiment
- generating evaluation reports
- selecting the best-performing model

---

# Responsibilities

Perform:

- Model evaluation
- Performance comparison
- Benchmarking
- Statistical validation
- Explainability evaluation
- Result reporting

Generate:

- Evaluation summary
- Benchmark comparison
- Performance tables
- Visualization recommendations
- Acceptance decision

---

# Evaluation Checklist

Evaluate using the metrics appropriate for the task.

Examples include:

- Accuracy
- Precision
- Recall
- F1 Score
- AUROC
- AUPRC
- Loss
- Calibration

For multi-label ECG classification, evaluate both overall and per-class
performance where applicable.

---

# Validation Checklist

Verify:

- Model converged successfully
- No obvious overfitting
- Validation metrics are consistent
- Test data was not used during training
- Results are reproducible
- Performance improvements are meaningful

---

# Workflow

For each evaluation:

1. Load experiment results.
2. Evaluate model performance.
3. Compare against previous baselines.
4. Validate experiment quality.
5. Generate evaluation summary.
6. Recommend whether the model should be accepted or rejected.

---

# Files to Maintain

Update when applicable:

.agents/project/

progress.md

research/experiment_journal.md

research/research_log.md

Do not modify project planning or architecture files.

---

# Rules

- Never modify model weights.
- Never retrain models.
- Use objective evaluation criteria.
- Compare against existing baselines whenever possible.
- Clearly distinguish measured results from observations.

---

# Success Criteria

The skill is successful when:

- Model performance is evaluated.
- Results are validated.
- Comparisons are documented.
- The best-performing model is identified.
- A clear recommendation is produced.