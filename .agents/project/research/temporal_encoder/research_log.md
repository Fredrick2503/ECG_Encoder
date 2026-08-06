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
  - Optimized the dataloader record exist check logic using `os.walk` to bypass slow sequential OneDrive disk access, reducing startup time by over **4500x** (from ~15m to <0.2s).
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
