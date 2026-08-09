# Search History — MasterMind Loop

This file records all strategy configurations that have been tried, along with
their outcomes. Used by @adaptive-trainer to avoid re-testing identical configs
and to guide mutation decisions.

---

## Pre-Loop Known Results (Reference)

| Config Hash | Architecture | Pretraining | Loss | LR | Dropout | ROC-AUC | F1 | Status |
|---|---|---|---|---|---|---|---|---|
| ref_001 | BiLSTM | reconstruction | BCE | 1e-3 | 0.2 | ~0.82 | — | Reference |
| ref_002 | BiLSTM | mae | BCE | 1e-3 | 0.2 | ~0.857 | — | Reference |
| ref_003 | ECGTransformer | mae | BCE | 3e-4 | 0.1 | ~0.872 | — | Reference |
| ref_004 | ECGResNet1D_SE | supervised | ASL | 3e-4 | 0.3 | — | — | Reference |
| ref_005 | Ensemble (ResNet+Trans) | — | ASL | — | — | 0.8918 | — | Best Known |

---

## MasterMind Loop Trials

*(Trial entries will be appended here by @adaptive-trainer after each trial.)*

---

*Maintained by @adaptive-trainer using the `adaptive-strategy-search` skill.*
