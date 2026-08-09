---
description: Template for notebook cell blocks appended after each experiment trial.
---

# Notebook Sync Template

Use this as a reference when @notebook-sync appends cells to a Jupyter notebook.
Each trial produces 3 cells appended to the end of the relevant notebook.

---

## Cell 1 — Markdown Header Cell

```markdown
---

## Trial <N> — <Architecture> | <Strategy> | <YYYY-MM-DD>

**MLflow Run ID:** `<run_id>`  
**Git Branch:** `<branch>`  
**Experiment:** `<experiment_name>`  
**Status:** ✅ Target Reached / ❌ Below Target / ⚠️ Barrier Detected
```

---

## Cell 2 — Markdown Metrics + Config Cell

```markdown
### Results

| Metric | This Trial | Previous Best | Baseline | Target |
|---|---|---|---|---|
| ROC-AUC (macro) | **<value>** | <prev> | <base> | ≥ 0.92 |
| Macro F1 | **<value>** | <prev> | <base> | — |
| Subset Accuracy | **<value>** | <prev> | <base> | — |
| Hamming Loss | **<value>** | <prev> | <base> | — |

### Configuration Snapshot

| Parameter | Value |
|---|---|
| Architecture | <arch> |
| Pretraining | <strategy> |
| Loss Function | <loss> |
| Learning Rate | <lr> |
| LR Schedule | <schedule> |
| Dropout | <value> |
| Weight Decay | <value> |
| Batch Size | <value> |
| Epochs (actual) | <N> |
| Training Time | <HH:MM:SS> |
```

---

## Cell 3 — Markdown Observations + Next Steps Cell

```markdown
### Observations

> <What happened during training. Key observations from the learning curves.>

### Barriers Detected

**Type:** <None / Overfitting / Underfitting / Gradient / Plateau / Class Imbalance>  
**Severity:** N/A / LOW / MEDIUM / HIGH / CRITICAL

> <Brief description of barrier or "No barriers detected.">

### Shortcomings

> <Honest limitations of this trial's approach.>

### Next Strategy

**Trial <N+1>:** <Architecture> + <Strategy>  
**Change:** <What is being modified and why>  
**Source:** <Barrier fix / Mutation / New architecture>

---
*Synced by @notebook-sync · MasterMind Loop Iteration <N> · <YYYY-MM-DD HH:MM>*
```

---

## Notebook JSON Patch (for programmatic use)

When updating a notebook file directly, each cell follows this structure:

```json
{
  "cell_type": "markdown",
  "metadata": {
    "mastermind_trial": "<TRIAL_ID>",
    "mastermind_iteration": <N>,
    "sync_date": "<YYYY-MM-DD>"
  },
  "source": [
    "<cell content lines as list>"
  ]
}
```

---

## Notes

- Always append cells — never insert or overwrite.
- Use `"cell_type": "markdown"` for all sync cells.
- Tag all sync cells with `mastermind_trial` metadata for traceability.
- If the notebook does not exist, create it with this standard header notebook structure.
