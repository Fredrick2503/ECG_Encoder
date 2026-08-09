---
name: notebook-sync
description: >
  Automatically update Jupyter notebooks with structured experiment result cells
  after every trial. Appends a new cell block documenting the trial config,
  metrics, outcomes, barriers, and next steps. Preserves all existing cells.
  Upgrades the existing sync-experiment-notebook skill with per-trial automation.
  Use after every experiment trial is logged.
risk: medium
source: project
---

# Notebook Sync

## Objective

Keep all project Jupyter notebooks synchronized with the latest experiment
results after every trial in the MasterMind loop.

This skill appends structured, readable cell blocks to the appropriate notebook
so that the notebooks serve as a living, accurate record of the research
process — not just a static tutorial.

---

## When to Use

Use this skill whenever:

- An experiment trial has completed and been logged
- A barrier report has been produced and fixed applied
- A new architecture or technique has been evaluated
- The MasterMind loop completes an iteration
- A manual sync is requested

---

## Responsibilities

- Identify the correct notebook(s) to update based on the trial's module
- Append a structured experiment summary cell block
- Add a metrics visualization cell (markdown table)
- Add a config snapshot cell (python comment block)
- Add a "next steps / barriers" cell
- Preserve all existing cells — never overwrite or delete
- Flag stale output cells from previous runs

---

## Notebook Map

| Module | Notebook |
|---|---|
| Data management + preprocessing | `notebooks/01_data_management_and_preprocessing.ipynb` |
| Temporal encoder experiments | `notebooks/02_temporal_representation_learning.ipynb` |
| Morphology encoder | `notebooks/03_morphology_encoder.ipynb` (create if needed) |
| Biomarker encoder | `notebooks/04_biomarker_encoder.ipynb` (create if needed) |
| Fusion + classification | `notebooks/05_fusion_and_classification.ipynb` (create if needed) |
| Ensemble evaluation | `notebooks/06_ensemble_evaluation.ipynb` (create if needed) |
| MasterMind loop overview | `notebooks/00_mastermind_overview.ipynb` (create if needed) |

---

## Cell Block Structure

Append the following block to the relevant notebook after each trial:

### Cell 1: Markdown header cell
```markdown
---
## Trial <N> — <Architecture> | <Strategy> | <Date>

**MLflow Run ID:** `<run_id>`  
**Branch:** `<git_branch>`  
**Status:** ✅ Success / ❌ Failed / ⚠️ Barrier Detected
```

### Cell 2: Markdown metrics table
```markdown
### Results

| Metric | Value | vs. Best |
|---|---|---|
| ROC-AUC | <value> | <delta> |
| Macro F1 | <value> | <delta> |
| Subset Accuracy | <value> | <delta> |
| Val Loss | <value> | <delta> |

### Config Snapshot
| Parameter | Value |
|---|---|
| Architecture | <arch> |
| Pretraining | <strategy> |
| Loss | <loss_fn> |
| LR Schedule | <schedule> |
| Dropout | <value> |
| Epochs | <N> |
```

### Cell 3: Markdown barriers + next steps
```markdown
### Barriers & Observations
<Barrier classification + brief description, or "None detected">

### Shortcomings
<Honest limitations of this trial>

### Next Strategy
<What the Adaptive Trainer will try next and why>
```

---

## Workflow

For every notebook sync cycle:

1. Read the trial result from `experiment_journal.md`.
2. Identify which notebook(s) to update based on the trial module.
3. Read the current notebook JSON.
4. Build the 3-cell block from the trial data.
5. Append the block to the end of the notebook's cell list.
6. Write the updated notebook JSON back to disk.
7. Log the sync to `mastermind_state.md`.

---

## Template Reference

Use the template at:
```
.agents/project/research/_templates/notebook_sync_template.md
```

---

## Rules

- **Never delete existing cells.** Only append.
- **Never re-run existing cells** — only add new cells with raw markdown or code.
- If the target notebook does not exist, create it with a standard header.
- Keep cell content concise — the notebook is a record, not a tutorial.
- Always include the MLflow run ID for cross-reference.
- Flag cells from failed runs with ❌ in the header.

---

## Success Criteria

The skill is successful when:

- Every trial has a corresponding cell block in the relevant notebook.
- No existing cells are overwritten or deleted.
- Metrics are accurate and cross-reference MLflow.
- Barriers and shortcomings are documented in the notebook.
- The notebook can be read chronologically as a research diary.
