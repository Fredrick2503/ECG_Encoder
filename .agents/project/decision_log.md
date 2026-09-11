# Decision Log

## Architecture Decisions

### DEC-001

**Decision:** Adopt a layered modular architecture.

**Status:** Accepted

**Reason:**
Separate concerns into independent modules that communicate through well-defined interfaces.

---

### DEC-002

**Decision:** Implement the system incrementally.

**Status:** Accepted

**Implementation Order:**

1. Data Management
2. Signal Preprocessing
3. Representation Generation
4. Temporal Encoder
5. Biomarker Encoder
6. Morphology Encoder

All remaining modules are deferred until this implementation is stable.

---

### DEC-003

**Decision:** Rewrite the entire codebase from scratch.

**Status:** Accepted

**Reason:**

Previous implementations serve only as architectural and implementation references.

No production code will be reused.

---

### DEC-004

**Decision:** Use a common ECGRecord domain model.

**Status:** Accepted

Every module exchanges ECGRecord objects instead of dataset-specific formats.

---

### DEC-005

**Decision:** Support multiple datasets through a common interface.

**Status:** Accepted

Initial datasets:

* PTB-XL
* MIT-BIH

Future datasets will plug into the same interface.

---

### DEC-006

**Decision:** Separate preprocessing from representation generation.

**Status:** Accepted

Preprocessing standardizes ECG signals.

Representation Generation creates:

* Temporal representation
* Biomarker representation
* Morphology representation

---

### DEC-007

**Decision:** Representation encoders must be independent.

**Status:** Accepted

Each encoder owns:

* representation learning
* training
* evaluation
* prediction
* explainability

---

### DEC-008

**Decision:** Maintain modular encoders before Multi-Modal Fusion Engine.

**Status:** Accepted

Encoders (Temporal, Biomarker, Morphology) produce standardized latent embeddings that interface with the downstream Fusion Engine.

---

### DEC-009

**Decision:** Use Continuous Wavelet Transform (CWT) for wave delineation over DWT.

**Status:** Accepted

**Reason:**

NeuroKit2's DWT delineator caused systematic boundary overestimation, inflating QRS duration to ~170 ms. CWT scale-space localization restores physiological boundaries (median QRS ~106 ms, median PR ~148 ms) and reduces QC warnings by 96.7%.

---

### DEC-010

**Decision:** Enforce strict patient-wise partitioning and leakage-free biomarker preprocessing.

**Status:** Accepted

**Reason:**

Prevent test-set information from leaking into training pipelines by fitting Imputer and Scaler exclusively on the patient-wise training split and transforming validation and test partitions.

---

### DEC-011

**Decision:** Multi-Task Regularized 32-D Latent Biomarker Encoders.

**Status:** Accepted

**Reason:**

Joint optimization of reconstruction loss (MSE) and multi-label diagnostic classification loss (BCEWithLogits) regularizes the 32-D latent space, increasing downstream Macro F1 by +6.09% over raw clinical features while preventing dimensional collapse (0/32 collapsed dims).
