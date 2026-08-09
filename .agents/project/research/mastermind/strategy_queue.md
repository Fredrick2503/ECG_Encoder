# Strategy Queue — MasterMind Loop

This file tracks pending and completed strategy configurations.

**Template format:** See `adaptive-strategy-search` skill for config schema.

---

## Pending Strategies

*(To be populated by @adaptive-trainer on first loop run.)*

Suggested initial queue (ordered by expected impact):

### Strategy 001 — ECGTransformer + MAE + WarmupCosine + ASL
```yaml
trial_id: trial_001
architecture: ECGTransformer
pretraining: mae
loss: AsymmetricLoss
lr_schedule: WarmupCosine
learning_rate: 2e-4
dropout: 0.2
weight_decay: 1e-4
augmentation: [random_noise, amplitude_scaling]
epochs: 60
batch_size: 64
source: initial
reason: "Best known single model is Transformer + ASL. Adding WarmupCosine LR and explicit augmentation to push past 0.89 plateau."
```

### Strategy 002 — ECGTransformer + Two-Stage (MAE → Supervised) + ASL
```yaml
trial_id: trial_002
architecture: ECGTransformer
pretraining: mae_then_supervised
loss: AsymmetricLoss
lr_schedule: CosineAnnealingLR
learning_rate: 1e-4
dropout: 0.3
weight_decay: 1e-4
augmentation: [random_noise, time_warping]
epochs: 80
batch_size: 32
source: initial
reason: "Two-stage pretraining may yield better feature initialization before supervised fine-tuning."
```

### Strategy 003 — ECGResNet1D+SE + Contrastive + FocalLoss
```yaml
trial_id: trial_003
architecture: ECGResNet1D_SE
pretraining: contrastive
loss: FocalLoss
focal_gamma: 2.0
lr_schedule: OneCycleLR
learning_rate: 3e-4
dropout: 0.25
weight_decay: 5e-5
augmentation: [amplitude_scaling, lead_dropout]
epochs: 50
batch_size: 64
source: initial
reason: "Contrastive pretraining on ResNet+SE with Focal Loss for class imbalance — unexplored combination."
```

---

## Completed Strategies

*(Entries will be moved here from Pending after each trial completes.)*

---

*Maintained by @adaptive-trainer using the `adaptive-strategy-search` skill.*
