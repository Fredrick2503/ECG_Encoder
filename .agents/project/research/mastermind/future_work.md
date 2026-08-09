# Future Work — MasterMind Loop

This file records research directions identified during the experiment loop
that are outside the current scope but worth pursuing.

---

## Known Pre-Loop Future Directions

### FW-001 — Spatial Lead Attention
Implement a spatial attention mechanism across ECG leads within the encoder
(e.g., cross-lead Transformer layer or lead-graph convolution). Current
architecture processes leads independently.

### FW-002 — Knowledge Distillation
Distill the best ensemble (ResNet+SE + ECGTransformer) into a single compact
model for deployment without performance loss.

### FW-003 — Self-Supervised Pretraining on Unlabeled ECG Data
PTB-XL has ~22K labeled records. Large unlabeled ECG corpora exist (e.g.,
MIMIC-IV ECG, ~800K records). Pre-training on unlabeled data before
fine-tuning on PTB-XL could significantly improve representation quality.

### FW-004 — Morphology Encoder Integration
The current pipeline uses only temporal features. Integrating the Morphology
Encoder (beat-level shape features) through the Fusion Engine is the next
major architectural milestone.

### FW-005 — Multi-Dataset Training
Training across PTB-XL + CPSC + MIT-BIH simultaneously (with dataset-specific
label alignment) could improve generalization and minority class performance.

### FW-006 — Clinical Validation
Model outputs have not been validated against clinical cardiologist annotations.
A prospective clinical validation study is required before any deployment
consideration.

---

## MasterMind Loop Future Work

### FW-007 — Systematic Filter-Encoder Grid Sweep (49 Combinations)
Systematically evaluate all combinations of the 7 signal filters (`none`, `bandpass`, `bandpass_notch`, `fir`, `wavelet`, `full_stack`, `robust_norm`) against the 7 core encoder architectures (`transformer`, `resnet_se`, `multiscale_cnn`, `bilstm`, `bigru`, `attn_bilstm`, `cnn_lstm_trans`). These trials are deferred to keep manual ablation sweeps focused on critical hyperparameter/regularization boundaries.

---

*Maintained by @thesis-doc using the `thesis-writer` skill.*
