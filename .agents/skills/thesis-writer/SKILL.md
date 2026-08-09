---
name: thesis-writer
description: >
  Generate and maintain thesis-quality documentation from structured experiment
  records. Updates thesis_notes.md, docs/, and research logs with experiment
  outcomes, barrier findings, shortcomings, methodology descriptions, and future
  directions. Produces academic prose suitable for a master's or doctoral thesis.
  Use after every experiment trial completes and after every barrier report.
risk: low
source: project
---

# Thesis Writer

## Objective

Continuously maintain thesis-ready documentation that accurately reflects every
experiment conducted, every finding observed, every barrier encountered, and
every shortcoming discovered during the research process.

This skill transforms raw experiment data into academic prose that can be
directly incorporated into the thesis manuscript with minimal editing.

---

## When to Use

Use this skill whenever:

- An experiment trial completes (success or failure)
- A barrier report has been produced
- A new technique or architecture has been evaluated
- A milestone is reached
- The MasterMind loop completes an iteration
- A comparison between trials is meaningful
- A shortcoming or limitation is identified

---

## Responsibilities

Maintain and update:

- `docs/<module>/thesis_notes.md` — per-module thesis notes
- `.agents/project/research/<feature>/thesis_notes.md` — experiment-level notes
- `docs/<module>/experiment_log.md` — chronological experiment log
- `docs/thesis/` — cross-module thesis draft sections

Write the following thesis sections as appropriate:

| Section | When to Write |
|---|---|
| **Methodology** | When a new architecture or technique is introduced |
| **Experimental Setup** | After data audit + config are finalized |
| **Results** | After trial completes with metrics |
| **Analysis & Discussion** | After comparing multiple trials |
| **Barriers & Limitations** | After every barrier report |
| **Shortcomings** | When a fix is impossible or target is unreachable |
| **Future Work** | After each iteration, note what could be tried next |

---

## Writing Style Rules

All generated documentation must:

1. **Be factual** — record what actually happened, not what was hoped for.
2. **Use academic prose** — complete sentences, formal register, no informal language.
3. **Be specific** — include exact metric values, epoch counts, config details.
4. **Distinguish** between observations, hypotheses, and conclusions.
5. **Cite techniques** — reference the original paper when a technique is used
   (e.g., "Focal Loss (Lin et al., 2017)").
6. **Record failures honestly** — a failed experiment is still a contribution.
7. **Use tables** for metric comparisons between trials.
8. **Use consistent notation** — ROC-AUC, Macro F1, Subset Accuracy throughout.

---

## Workflow

For every thesis update:

1. Read the trial result from `experiment_journal.md`.
2. Read any barrier report from `barriers/barrier_<trial_id>.md`.
3. Read the data audit from `trials/<trial_id>/data_audit.md`.
4. Write or append the appropriate thesis sections.
5. Update the chronological experiment log.
6. Update the shortcomings and limitations section.
7. Add future work suggestions if new ideas emerged.
8. Save all updated files.

---

## File Locations

```
docs/
  temporal_encoder/
    thesis_notes.md         ← primary thesis notes for temporal encoder
    experiment_log.md       ← chronological log of all trials
  preprocessing/
    thesis_notes.md
  data_management/
    thesis_notes.md

.agents/project/research/
  mastermind/
    thesis_notes.md         ← cross-trial thesis notes for MasterMind loop
    shortcomings.md         ← documented limitations and barriers
    future_work.md          ← future research directions
```

---

## Thesis Update Template Reference

Use the template at:
```
.agents/project/research/_templates/thesis_update_template.md
```

---

## Key Thesis Sections to Maintain

### Per-Trial Section Format

```markdown
### Trial <N> — <Architecture> + <Strategy>

**Date:** <date>
**Configuration:** <key hyperparameters>

**Results:**
| Metric | Value |
|---|---|
| ROC-AUC | <value> |
| Macro F1 | <value> |
| Subset Accuracy | <value> |

**Observations:**
<What was observed during training and evaluation.>

**Analysis:**
<Why the results are as they are, mechanistically.>

**Barriers Encountered:**
<Any barriers detected, with classification and proposed fix.>

**Shortcomings:**
<Honest limitations of this approach.>
```

---

## Rules

- Never delete previous thesis entries — always append.
- Never mark a result as better than it is — be precise.
- Always link to the MLflow run ID for reproducibility.
- Use consistent section headings across all thesis files.
- Shortcomings are a feature, not a bug — document them thoroughly.

---

## Success Criteria

The skill is successful when:

- Every trial has a thesis entry.
- Barriers and shortcomings are documented.
- The thesis notes are coherent and academically readable.
- Future work section is current.
- Tables accurately reflect all trial comparisons.
