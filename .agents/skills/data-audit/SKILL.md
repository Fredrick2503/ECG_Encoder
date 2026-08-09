---
name: data-audit
description: >
  Document the complete state of the ECG data pipeline for each experiment trial.
  Captures data source, version, split ratios, preprocessing steps applied,
  class distribution, quality flags, and any augmentation used. Produces a
  structured data_audit.md per experiment that is attached to the trial record.
  Use at the start of every experiment loop iteration.
risk: medium
source: project
---

# Data Audit

## Objective

Produce a complete, reproducible record of the data pipeline state at the time
of each experiment trial.

This skill ensures that every result can be traced back to the exact data that
produced it — including how the data was sourced, cleaned, split, and
preprocessed.

---

## When to Use

Use this skill at the **start of every experiment loop iteration**, before any
training begins.

Also use whenever:

- The dataset has changed or been updated
- Preprocessing configuration has been modified
- A new split or fold has been created
- Class balance or augmentation strategy has been changed

---

## Responsibilities

Document and audit:

- Dataset identity (name, version, source, download date)
- Record counts (total, train, validation, test)
- Class distribution (per split, label frequency table)
- Applied preprocessing steps and their parameters
- Applied augmentation techniques and their probabilities
- Known data quality issues (noisy records, missing leads, label noise)
- Data integrity checksums (if available)
- Split strategy (stratified k-fold, random, patient-level, etc.)
- Any exclusions or filters applied

---

## Workflow

For every experiment trial:

1. Read the current dataset configuration from `config/` or training script.
2. Load and count records per split.
3. Compute class distribution per split.
4. List all preprocessing steps from the preprocessing pipeline config.
5. List augmentation strategies and their parameters.
6. Flag any known quality issues.
7. Write the structured `data_audit.md` to the trial's research directory.
8. Attach the audit path to the experiment trial record.

---

## Output File

Write to:
```
.agents/project/research/<feature_name>/trials/<trial_id>/data_audit.md
```

Also summarize key stats in the experiment trial record for cross-reference.

---

## Data Audit Template Reference

Use the template at:
```
.agents/project/research/_templates/data_audit_template.md
```

---

## Key Fields to Always Record

| Field | Description |
|---|---|
| `dataset_name` | PTB-XL, MIT-BIH, CPSC, etc. |
| `dataset_version` | Commit hash or download date |
| `total_records` | Integer count |
| `train_count` | Count of training records |
| `val_count` | Count of validation records |
| `test_count` | Count of test records |
| `split_strategy` | How splits were made (e.g., stratified 10-fold) |
| `preprocessing_steps` | Ordered list of applied transforms |
| `augmentation` | List of augmentations with probabilities |
| `class_distribution` | Per-label counts for each split |
| `quality_flags` | Known issues or exclusions |
| `sampling_rate` | ECG signal sampling rate in Hz |
| `lead_configuration` | Number of leads used |

---

## Rules

- Never assume data is unchanged — always re-audit at each trial start.
- Never modify raw dataset files during auditing.
- Record the preprocessing config exactly as applied, not as intended.
- Flag discrepancies between expected and actual record counts.
- Preserve all audit files — they are part of the experiment record.

---

## Success Criteria

The skill is successful when:

- A complete `data_audit.md` exists for every trial.
- Class distributions are documented per split.
- Preprocessing steps are listed exactly as applied.
- Any data quality flags are recorded.
- The audit is attached to the experiment trial record.
