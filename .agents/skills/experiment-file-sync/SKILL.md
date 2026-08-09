---
name: experiment-file-sync
description: >
  Synchronize all experiment artifacts, logs, reports, and documentation across
  the project's output directories after every trial. Ensures that outputs/,
  docs/, notebooks/, and .agents/project/research/ are all consistent and
  up to date. Use after every trial is logged and after every notebook sync.
risk: medium
source: project
---

# Experiment File Sync

## Objective

Maintain a single, consistent view of all experiment artifacts across the
project filesystem after every trial.

This skill is the "finisher" of the experiment loop — it ensures nothing is
left in a stale or inconsistent state. After training runs, logging, and
documentation are complete, this skill performs a structured sync that
propagates all changes to the correct locations.

---

## When to Use

Use this skill whenever:

- An experiment trial has been logged
- A barrier report has been written
- A thesis update has been made
- The MasterMind loop completes an iteration
- The `/sync-workspace` workflow is invoked
- Before a git commit on the main branch

---

## Responsibilities

Sync and organize:

- `outputs/reports/` — consolidated experiment comparison reports
- `outputs/plots/` — training curves, metric plots per trial
- `docs/<module>/` — per-module documentation
- `notebooks/` — ensure notebooks reflect latest state
- `.agents/project/research/` — all structured research files
- `mlruns/` — confirm MLflow artifacts are intact

---

## Sync Checklist

For every sync cycle, verify and update:

### `outputs/reports/`
- [ ] `experiments_comparison_report.md` updated with latest trial
- [ ] Per-trial report file exists: `trial_<N>_report.md`
- [ ] Ensemble evaluation report updated (if ensemble was run)

### `outputs/plots/`
- [ ] Training loss curve saved for latest trial
- [ ] Validation AUC curve saved for latest trial
- [ ] Per-class F1 bar chart saved (if per-class metrics computed)

### `docs/<module>/`
- [ ] `thesis_notes.md` updated
- [ ] `experiment_log.md` updated
- [ ] Any new architecture diagrams or pipeline diagrams added

### `notebooks/`
- [ ] Latest notebook reflects current experiment state
- [ ] No stale output cells from old runs remain unflagged
- [ ] Experiment summary cell present and up to date

### `.agents/project/research/`
- [ ] `experiment_journal.md` updated
- [ ] `mastermind_state.md` updated
- [ ] `search_history.md` updated
- [ ] Latest barrier report present (if applicable)
- [ ] `strategy_queue.md` reflects current pending strategies

### Git Status
- [ ] All new files are staged
- [ ] Commit message prepared summarizing the trial

---

## Workflow

For every sync cycle:

1. Read the latest trial ID from `mastermind_state.md`.
2. Check each item in the sync checklist.
3. For missing items: create stubs or flag as TODO.
4. For stale items: update with latest data.
5. Confirm `outputs/reports/experiments_comparison_report.md` is current.
6. Confirm all research files are up to date.
7. Log the sync completion to `mastermind_state.md`.
8. Optionally: stage all changes with `git add -A` and prepare commit message.

---

## File Reference

```
outputs/
  reports/
    experiments_comparison_report.md   ← ALWAYS update
    trial_<N>_report.md                ← one per trial
  plots/
    trial_<N>_loss.png
    trial_<N>_auc.png

docs/
  temporal_encoder/
    thesis_notes.md
    experiment_log.md

notebooks/
  02_temporal_representation_learning.ipynb

.agents/project/research/
  mastermind/
    mastermind_state.md
    experiment_journal.md
    strategy_queue.md
    search_history.md
    barriers/
      barrier_<trial_id>.md
```

---

## Rules

- Never delete existing output files without explicit user approval.
- Always update `experiments_comparison_report.md` — it is the single source of truth for experiment comparisons.
- Flag (do not fix) inconsistencies that require human review.
- Keep sync logs brief but complete.
- Do not modify raw data files.

---

## Success Criteria

The skill is successful when:

- All output files reflect the latest trial.
- The comparison report is current.
- Research files are consistent.
- No stale artifacts remain unflagged.
- Git status is clean or staged for commit.
