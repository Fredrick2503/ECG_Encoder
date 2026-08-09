---
name: barrier-detection
description: >
  Detect, classify, and resolve training and performance barriers in the ECG
  experiment pipeline. When a trial fails to meet target metrics or exhibits
  pathological training behavior (overfitting, gradient collapse, loss explosion,
  plateau), this skill diagnoses the root cause, searches arXiv/web for modern
  solutions, and returns an actionable barrier report with specific fixes.
  Use whenever a trial outcome is below threshold or shows warning signs.
risk: high
source: project
---

# Barrier Detection

## Objective

Identify exactly WHY a training trial underperformed and WHAT to do about it.

This skill is the diagnostic and research engine of the MasterMind system.
It bridges the gap between "the model isn't good enough" and "here is a
specific, literature-backed fix to try next."

---

## When to Use

Use this skill whenever:

- Trial ROC-AUC is below the target threshold
- Validation loss is diverging from training loss (overfitting)
- Training loss is not decreasing (underfitting or learning rate issue)
- Gradient norms are abnormal (vanishing or exploding)
- Class-level F1 scores are severely imbalanced
- A training run crashed unexpectedly
- The search is plateauing across multiple consecutive trials

---

## Barrier Classification

Classify every barrier into one of the following categories:

| Barrier Type | Signals |
|---|---|
| **Overfitting** | val_loss >> train_loss, val_AUC drops after early peak |
| **Underfitting** | Both losses high, AUC never improves beyond 0.6 |
| **Gradient Collapse** | Loss = NaN, gradient norm → 0 |
| **Gradient Explosion** | Loss = NaN, gradient norm → ∞ |
| **Loss Plateau** | Loss stagnant for 10+ epochs |
| **Class Imbalance Failure** | Per-class F1 variance > 0.4 |
| **Data Quality Issue** | Unexpected NaN in inputs, zero-variance signals |
| **Architecture Mismatch** | Model too shallow/deep for signal length |
| **Learning Rate Issue** | Loss oscillates or never converges |
| **Strategy Mismatch** | Pretraining strategy hurts fine-tuning |

---

## Diagnostic Workflow

For every barrier detection cycle:

1. **Read trial result** from `experiment_journal.md`:
   - Final and per-epoch metrics (loss, AUC, F1, subset_acc)
   - Gradient norms (if logged)
   - Per-class metrics
   - Training config

2. **Classify the barrier** using the classification table above.

3. **Search for solutions** using:
   - `search_web` with query: `ECG deep learning <barrier_type> solution`
   - `literature_search_arxiv` with query: `ECG representation learning <barrier_type>`
   - Focus on: last 3 years, self-supervised learning, time-series classification

4. **Formulate fixes** — map literature findings to concrete config changes:
   - Specific parameter values to change
   - New techniques to add (e.g., gradient clipping, mixup, label smoothing)
   - Architecture changes to try
   - Loss function alternatives

5. **Write barrier report** to:
   ```
   .agents/project/research/mastermind/barriers/barrier_<trial_id>.md
   ```

6. **Return structured fix** to MasterMind for dispatch to `@adaptive-trainer`.

---

## Barrier Report Format

```markdown
# Barrier Report — Trial <trial_id>

## Barrier Classification
Type: <barrier_type>
Severity: [CRITICAL | HIGH | MEDIUM | LOW]
Confidence: [HIGH | MEDIUM | LOW]

## Evidence
- train_loss: <value>
- val_loss: <value>
- roc_auc: <value>
- macro_f1: <value>
- Observation: <what was observed>

## Root Cause Analysis
<Explain why this is happening, mechanistically>

## Literature Findings
- Paper: <title>, <year>, <link>
  Finding: <what they found and what fixed it>
- ...

## Proposed Fixes (ordered by priority)
1. <Fix 1> — rationale: <why>
2. <Fix 2> — rationale: <why>
3. <Fix 3> — rationale: <why>

## Override Config for Next Trial
```yaml
<specific config fields to override>
```

## Shortcomings Documented
<Honest assessment of limitations that may not be fixable>

## Future Research Directions
<Ideas for future investigation beyond this experiment cycle>
```

---

## Escalation Conditions

Report to MasterMind for user escalation if:

- The same barrier type has been encountered ≥ 3 consecutive trials
- All proposed fixes have been tried without improvement
- The barrier is caused by a fundamental data limitation (e.g., insufficient
  labeled data for a specific pathology)
- The target metric is unreachable given the current architecture family

---

## Rules

- Always search literature before proposing fixes. Do not rely solely on
  heuristics — ground every recommendation in evidence.
- Record ALL barriers, not just blocking ones. Even minor barriers are
  research data.
- Never delete a barrier report — they form the shortcomings section of the
  thesis.
- Be honest about shortcomings — do not oversell fixes.
- Distinguish between fixable barriers and fundamental limitations.

---

## Success Criteria

The skill is successful when:

- Every underperforming trial has a barrier report.
- Every barrier is classified with evidence.
- Literature-backed fixes are proposed.
- The structured fix config is correctly passed to `@adaptive-trainer`.
- Escalation happens promptly when stuck.
- All shortcomings are documented for thesis use.
