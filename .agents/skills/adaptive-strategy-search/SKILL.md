---
name: adaptive-strategy-search
description: >
  Manage the autonomous search over training strategies for the ECG models.
  Maintains a strategy queue, mutates configurations based on trial outcomes
  and barrier reports, tracks the full search history, and proposes the next
  best configuration to attempt. Use whenever a trial completes and a new
  strategy is needed, or when a barrier fix must be integrated into the next run.
risk: high
source: project
---

# Adaptive Strategy Search

## Objective

Continuously evolve training strategies to find configurations that achieve
target performance metrics.

This skill maintains the full search history, applies intelligent mutations to
previously-tried configurations, integrates barrier analyst recommendations,
and proposes the next configuration for the MasterMind to dispatch.

---

## When to Use

Use this skill whenever:

- A trial has completed and the next strategy must be selected
- A barrier report has been received and a corrective strategy must be built
- The strategy queue needs to be initialized for a new experiment loop
- A performance plateau requires a more aggressive configuration change
- A new architecture or technique should be tested

---

## Responsibilities

Manage:

- Strategy queue (pending configurations)
- Search history (all tried configurations + their outcomes)
- Mutation rules (how to evolve a config from one trial to the next)
- Barrier integration (apply barrier fixes into the next config)
- Convergence detection (recognize when search has plateaued)
- Strategy diversity enforcement (avoid re-testing the same configs)

---

## Strategy Dimensions

The search space covers the following axes:

### Architecture
- `BiLSTM` (baseline)
- `ECGTransformer`
- `ECGMultiScaleCNN`
- `ECGResNet1D` (with/without SE attention)
- `ECGTransformer` + SE hybrid

### Pretraining Strategy
- `reconstruction` (autoencoder)
- `mae` (masked autoencoder)
- `contrastive` (SimCLR / InfoNCE)
- `supervised_only`
- `mae` → `supervised` (two-stage)

### Loss Function
- `BCEWithLogitsLoss`
- `FocalLoss` (gamma sweep: 0.5, 1.0, 2.0, 3.0)
- `AsymmetricLoss` (ASL)
- `FocalLoss + ASL` hybrid

### Learning Rate Schedule
- `ReduceLROnPlateau`
- `CosineAnnealingLR`
- `OneCycleLR`
- `WarmupCosine`

### Regularization
- Dropout: [0.1, 0.2, 0.3, 0.4, 0.5]
- Weight Decay: [1e-5, 1e-4, 1e-3]

### Data Augmentation
- None
- Random noise
- Time warping
- Lead dropout
- Amplitude scaling

---

## Mutation Rules

Apply the following when generating the next strategy from a completed trial:

| Situation | Mutation |
|---|---|
| Overfitting | Increase dropout, increase weight decay, reduce LR |
| Underfitting | Decrease dropout, increase model capacity, increase LR |
| Loss plateau | Switch loss function, try OneCycleLR |
| Low F1 (imbalanced) | Switch to ASL or Focal loss, increase augmentation |
| Gradient collapse | Add pre-layer norm, reduce LR, switch to Transformer |
| Slow convergence | Switch to OneCycleLR or WarmupCosine |
| Barrier fix received | Apply exact fix from barrier report as overrides |

---

## Workflow

For every strategy selection cycle:

1. Read `mastermind_state.md` to get:
   - Last trial outcome (metrics, barriers, shortcomings)
   - Search history (all tried configs)
   - Any barrier fix from `@barrier-analyst`
2. Identify what went wrong in the last trial.
3. Apply the appropriate mutation rule(s).
4. Check the new config against the search history — avoid exact duplicates.
5. Apply any barrier fix overrides.
6. Write the new config to `strategy_queue.md`.
7. Return the config to MasterMind for dispatch.

---

## State Files

Maintain:

```
.agents/project/research/mastermind/
  strategy_queue.md       ← pending configurations
  search_history.md       ← all tried configs + outcomes
  convergence_log.md      ← plateau detection history
```

---

## Strategy Config Format

Each strategy is a structured block:

```yaml
trial_id: trial_<N>
architecture: ECGTransformer
pretraining: mae
loss: AsymmetricLoss
lr_schedule: WarmupCosine
learning_rate: 3e-4
dropout: 0.3
weight_decay: 1e-4
augmentation: [random_noise, amplitude_scaling]
epochs: 50
batch_size: 64
source: mutation_from_trial_<N-1>
reason: "Overfitting detected — increased dropout + weight decay"
```

---

## Rules

- Never repeat an identical configuration.
- Always record the reason for each mutation.
- Barrier fixes take priority over generic mutation rules.
- If the search space is exhausted, report to MasterMind for user escalation.
- Diversity: at least one axis must change between consecutive trials.

---

## Success Criteria

The skill is successful when:

- Every trial has a clearly reasoned next configuration.
- The search history is complete and non-redundant.
- Barrier fixes are correctly integrated into strategies.
- Convergence is detected and reported.
- The best configuration is clearly identified in the search history.
