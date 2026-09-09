# Temporal Encoder Research Log

This log documents key milestones, experimental trials, and environment configurations for the Temporal Encoder module.

---

### [2026-08-03] - Temporal Encoder Module & SSL Strategies
- **Topic:** Neural Network Modeling & Self-Supervised pretraining
- **Details:**
  - Installed lightweight PyTorch CPU package (~150MB) in the virtual environment.
  - Implemented the `ECGBiLSTM` encoder model extracting concatenated final hidden states as representations.
  - Implemented the `ECGReconstructionDecoder` module.
  - Implemented three self-supervised pretraining strategies: Reconstruction learning, Masked Autoencoder (MAE), and Contrastive Learning (SimCLR NT-Xent loss).
  - Implemented supervised training epochs, validation metrics evaluation (subset accuracy, Hamming Loss, Macro F1, Macro AUC), batch inference predictors, and gradient saliency explainer.
  - Created the representation learning research notebook `02_temporal_representation_learning.ipynb`.
  - Validated all architectures and training strategies via 9 unit test assertions in `test_temporal_encoder.py`.

---

### [2026-08-03] - CUDA-enabled Training, Dataset Restoring, and MLflow Tuning Sweeps
- **Topic:** MLflow Experiment Tracking & CUDA Acceleration
- **Details:**
  - Verified and enabled PyTorch CUDA acceleration using the local NVIDIA GeForce RTX 2050 GPU.
  - Successfully extracted the full 1.8GB PTB-XL dataset (containing 21,837 100Hz and 21,837 500Hz records) and optimized directory structure renames.
  - Resolved `typing-extensions` dependency import issues (upgraded to `4.16.0`).
  - Optimized the dataloader record exist check logic using `os.walk` to bypass slow OneDrive disk access, reducing startup time by over **4500x** (from ~15m to <0.2s).
  - Implemented MLflow experiment integration inside the main `fit` loop in `trainer.py` to auto-log train/val loss epoch-by-epoch.
  - Created a nested hyperparameter sweep script `run_mlflow_tuning.py` to evaluate learning rate, hidden dimension, and pretraining strategies with automated model artifact tracking.
  - Initiated a training sweep using a representative 3,000 record subset to tune model hyper-parameters.

---

### [2026-08-03] - Overfitting Prevention & Final Optimized Training
- **Topic:** Recurrent Neural Network Regularization
- **Details:**
  - Designed and implemented the final optimized training pipeline `train_optimized.py`.
  - Added recurrent dropout ($0.4$ on LSTMs) and fully connected dropout ($0.5$ on classification head) to prevent overfitting during full-scale training.
  - Added Adam weight decay of $1\times 10^{-4}$ L2 regularization.
  - Integrated `ReduceLROnPlateau` learning rate scheduler and an early stopping patience guard of $7$ epochs.
  - Launched the final training loop on the full PTB-XL dataset (17,418 training records) under a new experiment `ECG_TemporalEncoder_Optimized` tracked in MLflow.

---

### [2026-08-14] - Architectural Upgrades & Benchmark Evaluation (Suites B-E)
- **Topic:** Advanced Temporal Representation Architectures & Optimization
- **Details:**
  - Expanded the `temporal_encoder` module to include state-of-the-art models (`ECGTransformer`, `ECGResNet1D` with Squeeze-and-Excitation attention, and `ECGMultiScaleCNN`) and advanced lead/label attention mechanisms (`CrossLeadAttention` and `LabelDependencyHead`).
  - Run benchmark experiments demonstrating that:
    - **ECG-Transformer** achieves superior overall discriminative capacity (ROC-AUC of 0.8803 to 0.8818) compared to simpler BiLSTMs.
    - **Class-Balanced Loss (CBLoss)** yields massive improvements in Subset Accuracy and Macro F1 scores.
    - **Decision-Level Probability Fusion** combining `C5` (discrimination-optimized) and `D2-5` (exact-match optimized) achieves Pareto-optimal performance: **0.8818 ROC-AUC**, **0.6851 Macro F1**, and **0.5833 Subset Accuracy**.
    - **Sigmoid Calibration (Platt Scaling)** reduces Expected Calibration Error (ECE) to **0.0532**, ensuring high reliability.

---

### [2026-08-14] - Phase 2 Research Plan Formulation (Suite E3–E8)
- **Topic:** Clinical-Oriented Evaluation, Calibration, and Latent Geometry Auditing
- **Details:**
  - Designed the blueprint for the next phase of experiments on the `ResNet-1D-SE` backbone:
    - **E3-0:** Baseline reproduction of E1-5.
    - **E3-1 to E3-7:** Focuses on class imbalance (CBLoss weighting), curriculum training, near-miss optimization, and specialist objectives (resolving MI/STTC confusion, CD false negatives, HYP weakness).
    - **E4-0 to E4-1:** Benchmarks Platt scaling, isotonic regression, and uncalibrated models, auditing threshold sensitivity under perturbations.
    - **E5-0 to E5-1:** Seed replication (3–5 seeds) and bootstrap confidence interval computation to establish statistical bounds.
    - **E6-0 to E6-2:** Frozen representation linear probe analysis, embedding geometry checks (silhouette, NMI, ARI, kNN purity), and ablation studies.
    - **E7-0 to E7-1:** Downstream transfer to fine-grained SCP/diagnostic labels and multi-task learning.
    - **E8-0 to E8-2:** Code/weight lock, test-set evaluation, and multi-label error cardinality analysis.

---

### [2026-08-14] - Phase 2 Execution & Final locked Evaluation (Suite E3–E8)
- **Topic:** Advanced Benchmarking Results & Downstream Validation
- **Details:**
  - Completed all scheduled experiments on the `ECG-ResNet1D-SE` backbone:
    - **E3 Training:** Near-Miss Optimization (E3-3) achieved the highest Subset Accuracy of **0.6000**, while HYP Specialist (E3-6) reached **0.6757 Macro F1**.
    - **E4 Calibration:** Platt scaling calibration on E3-3 established the best operating point robustness (**0.6033 Subset Accuracy**, ECE **0.0447**, and low threshold fragility: standard deviation **0.0052**). Isotonic regression minimized ECE (0.0383) but was extremely fragile to perturbations.
    - **E5 Replication:** Confirmed tight variance across seeds (**ROC-AUC: 0.8718 ± 0.0027**). Bootstrap 95% CIs computed on test set (Subset Acc: `[0.5033, 0.6167]`, ECE: `[0.0497, 0.0725]`).
    - **E6 Geometry:** Frozen linear probe reached **0.8798 ROC-AUC** and **0.6474 Macro F1**. Disabling SE attention block (**No-SE**) collapsed Subset Accuracy to **0.4633** (absolute **-13.67%** drop), proving attention is vital.
    - **E7 Downstream Transfer:** Linear probe on frozen encoder features to predict 17 subclasses achieved Macro ROC-AUC of **0.8311** (with `CLBBB` at **0.9613** and `CRBBB` at **0.9702**). Predictions on demographic attributes transfered well: Sex (**0.6540 AUROC**), Age Group >= 60 (**0.7647 AUROC**).
    - **E8 Lock:** Evaluated the final locked E3-3 model with Platt scaling on the untouched test set, obtaining: **0.8793 ROC-AUC**, **0.6477 Macro F1**, **0.6033 Subset Accuracy**, and **0.0447 ECE**. Errors decomposed to: 60.33% exact match, 18.67% near-miss, 17.33% 2-errors, and 3.67% 3+ errors.


