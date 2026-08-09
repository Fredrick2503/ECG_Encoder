---
description: Template for recording a complete experiment trial in the journal.
---

# Experiment Trial <TRIAL_ID>

**Date:** <YYYY-MM-DD HH:MM>  
**Loop Iteration:** <N>  
**Git Branch:** <branch_name>  
**MLflow Run ID:** `<run_id>`  
**MLflow Experiment:** <experiment_name>  
**Status:** ✅ Success / ❌ Failed / ⚠️ Barrier Detected / 🔄 In Progress  

---

## Configuration

| Parameter | Value |
|---|---|
| Architecture | <BiLSTM / ECGTransformer / ECGResNet1D / ECGMultiScaleCNN> |
| Pretraining Strategy | <reconstruction / mae / contrastive / supervised_only> |
| Loss Function | <BCEWithLogitsLoss / FocalLoss / AsymmetricLoss> |
| Optimizer | <Adam / AdamW / SGD> |
| Learning Rate | <value> |
| LR Schedule | <ReduceLROnPlateau / CosineAnnealingLR / OneCycleLR> |
| Dropout | <value> |
| Weight Decay | <value> |
| Batch Size | <value> |
| Epochs (planned) | <N> |
| Epochs (actual) | <N> |
| Early Stopping Patience | <N> |
| Focal Loss Gamma | <value or N/A> |
| Augmentation | <list or None> |

**Config Source:** Mutation from Trial <N-1> / Initial / Barrier Fix from Trial <N>

**Mutation Reason:**  
> <Why this config was chosen. What was changed from the previous trial and why.>

---

## Data Audit Reference

**Data Audit File:** `trials/<TRIAL_ID>/data_audit.md`  
**Dataset:** PTB-XL full (21,837 records) / subset  
**Split:** Fold <N>, stratified 10-fold  
**Key Flags:** <any quality flags from data audit>

---

## Training Dynamics

| Epoch | Train Loss | Val Loss | Val AUC | Val F1 |
|---|---|---|---|---|
| 1 | | | | |
| 10 | | | | |
| 25 | | | | |
| Best | | | | |
| Final | | | | |

**Best Epoch:** <N>  
**Convergence:** Smooth / Oscillating / Plateau / Diverging  
**Training Time:** <HH:MM:SS>

---

## Final Test Metrics

| Metric | Value | vs. Previous Best | vs. Baseline |
|---|---|---|---|
| ROC-AUC (macro) | <value> | <delta> | <delta> |
| Macro F1 | <value> | <delta> | <delta> |
| Subset Accuracy | <value> | <delta> | <delta> |
| Hamming Loss | <value> | <delta> | <delta> |

**Target ROC-AUC:** 0.92  
**Gap to Target:** <value>  
**Result:** ✅ Target Reached / ❌ Below Target

---

## Per-Class Metrics

| Class | AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| NORM | | | | |
| MI | | | | |
| STTC | | | | |
| CD | | | | |
| HYP | | | | |

**Class Imbalance Impact:** <describe if any classes are significantly harder>

---

## Observations

> <What happened during training? What was notable? What did the metrics show epoch-by-epoch?>

---

## Barriers Detected

**Barrier Type:** <None / Overfitting / Underfitting / Gradient Issue / Plateau / Other>  
**Barrier Severity:** N/A / LOW / MEDIUM / HIGH / CRITICAL  
**Barrier Report:** `barriers/barrier_<TRIAL_ID>.md` / N/A

> <Brief description of any barriers encountered.>

---

## Shortcomings

> <Honest limitations of this trial. What this architecture cannot do. What the data cannot support. What the strategy fails to address.>

---

## Artifacts

| Artifact | Path |
|---|---|
| Best checkpoint | `models/<TRIAL_ID>_best.pt` |
| MLflow artifacts | `mlruns/<experiment_id>/<run_id>/artifacts/` |
| Training log | `logs/<TRIAL_ID>_train.log` |
| Trial report | `outputs/reports/trial_<TRIAL_ID>_report.md` |

---

## Next Action

**Next Strategy:** Trial <N+1> — <Architecture> + <Strategy>  
**Reason:** <What will change and why>

---

*Recorded by @experiment-logger using the `experiment-file-sync` skill.*
