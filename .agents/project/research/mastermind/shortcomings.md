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
**Partially mitigated by:** Asymmetric Loss (ASL), Focal Loss, Class-Balanced Loss (CBLoss), Hard-Minority Sampling  
**Residual gap:** Per-class F1 variance remains high; HYP F1 at 0.5818 in best model  
**Thesis note:** Class imbalance in PTB-XL is a well-documented challenge
(Wagner et al., 2020) that limits achievable Macro F1 without external data.

### SL-002 — Lead-Independent Processing
**Date:** Pre-loop  
**Type:** Architecture Limitation  
**Severity:** LOW-MEDIUM  
**Description:** The current temporal encoder processes each ECG lead
independently before aggregation. This may miss inter-lead spatial relationships
(e.g., axis deviation patterns visible only through multi-lead comparison).  
**Attempted:** Cross-Lead Attention (C16/C17) — failed catastrophically on 2K (AUC dropped to 0.67-0.69).  
**Future work:** Spatial attention across leads requires significantly more data (>10K records).

### SL-003 — Ensemble Complexity vs. Single Model
**Date:** Pre-loop  
**Type:** Deployment Limitation  
**Severity:** LOW  
**Description:** The best current result (E1-8) requires a probability fusion of
two ResNet-SE models (C5 and D2-5), approximately 2× the inference cost of a
single model.  
**Future work:** Knowledge distillation from ensemble into a single model.

---

## MasterMind Loop Shortcomings

### SL-004 — 2K Training Subset Generalization Gap
**Date:** 2026-08-14  
**Type:** Data Limitation  
**Severity:** HIGH  
**Discovered in:** All suites B–E2  
**Description:** All experiments were conducted on a 2,000-record PTB-XL subset.
This restricts model capacity and representation generalization. Full PTB-XL
has 21,837 records — scaling to full data is expected to significantly improve
both ROC-AUC and Macro F1, potentially closing the gap to target metrics.  
**Status:** Unresolved — next priority.

### SL-005 — Weak kNN Retrieval Performance
**Date:** 2026-08-14  
**Type:** Representation Quality Limitation  
**Severity:** MEDIUM  
**Discovered in:** Suite E2 (E2-3-C5, E2-3-D2, E2-3-Joint)  
**Description:** kNN classification AUC ranges from 0.45–0.55, well below
the linear probe AUC (0.62–0.73). This indicates that the representation space
is not locally organized enough for non-parametric nearest-neighbor retrieval.
Supervised training objectives do not impose metric learning structure on the
embedding space.  
**Future work:** Self-supervised pretraining (MAE, SimCLR) on full PTB-XL before
supervised fine-tuning. Triplet loss or Prototypical Networks for metric learning.

### SL-006 — Joint Concatenated Subset Accuracy Collapse
**Date:** 2026-08-14  
**Type:** Representation Quality Limitation  
**Severity:** MEDIUM  
**Discovered in:** Suite E2 (E2-2)  
**Description:** The joint concatenated representation achieves the best linear
probe ROC-AUC (0.7331) and Macro F1 (0.5362), but Subset Accuracy collapses
to 0.0367. This indicates that the linear classification head cannot capture
the multi-label joint structure needed for exact-match predictions when
representations are concatenated from two models trained under different loss
objectives. The head learns marginal class probabilities, not joint label
distributions.  
**Future work:** Use multi-label structured prediction heads (e.g., label-graph
neural networks) instead of a simple linear probe for joint representations.

### SL-007 — No Clinical Cardiologist Validation
**Date:** 2026-08-14  
**Type:** Clinical Validity Limitation  
**Severity:** HIGH  
**Description:** Model outputs have not been validated against clinical
cardiologist annotations on novel ECG records outside PTB-XL. All evaluation
is conducted on PTB-XL test splits, which use automated annotation labels.  
**Implication:** Results cannot be directly cited as clinically validated
performance metrics without a prospective validation study.  
**Future work:** Required before any deployment consideration.

---

*Maintained by @barrier-analyst and @thesis-doc. Last updated: 2026-08-14.*
