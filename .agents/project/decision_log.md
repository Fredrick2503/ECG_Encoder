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
5. Morphology Encoder

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

**Decision:** Freeze project scope after the Morphology Encoder.

**Status:** Accepted

The following components remain planned only:

* Biomarker Encoder
* Fusion Engine
* Unified Classification
* Explainability
* Deployment

They will not be designed or implemented until the roadmap is officially extended.
