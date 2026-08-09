# Shortcomings & Limitations — MasterMind Loop

This file documents all barriers encountered and limitations discovered during
the MasterMind experiment loop. It forms the basis of the Limitations and
Shortcomings chapter of the thesis.

---

## Known Pre-Loop Limitations

### SL-001 — Class Imbalance (PTB-XL)
**Date:** Pre-loop  
**Type:** Data Limitation  
**Severity:** MEDIUM  
**Description:** PTB-XL exhibits significant label imbalance. NORM dominates
(~29%), while HYP and MI are underrepresented. This systematically depresses
per-class F1 for minority classes regardless of model architecture.  
**Partially mitigated by:** Asymmetric Loss (ASL), Focal Loss  
**Residual gap:** Per-class F1 variance remains high (~0.3–0.4)  
**Thesis note:** Class imbalance in PTB-XL is a well-documented challenge
(Wagner et al., 2020) that limits achievable Macro F1 without external data.

### SL-002 — Lead-Independent Processing
**Date:** Pre-loop  
**Type:** Architecture Limitation  
**Severity:** LOW-MEDIUM  
**Description:** The current temporal encoder processes each ECG lead
independently before aggregation. This may miss inter-lead spatial relationships
(e.g., axis deviation patterns visible only through multi-lead comparison).  
**Future work:** Spatial attention across leads (e.g., lead-level transformer).

### SL-003 — Ensemble Complexity vs. Single Model
**Date:** Pre-loop  
**Type:** Deployment Limitation  
**Severity:** LOW  
**Description:** The best current result (ROC-AUC 0.8918) requires an ensemble
of ResNet+SE and ECGTransformer, approximately 2× the inference cost of a
single model.  
**Future work:** Knowledge distillation from ensemble into a single model.

---

## MasterMind Loop Shortcomings

*(Entries will be appended here by @barrier-analyst and @thesis-doc as trials proceed.)*

---

*Maintained by @barrier-analyst and @thesis-doc.*
