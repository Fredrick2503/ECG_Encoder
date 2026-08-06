# Temporal Encoder Experiment Journal

This journal documents the design, configurations, and outcomes of experiments run on the Temporal Encoder module.

---

## [2026-08-03] - Temporal Encoder Hyperparameter Tuning Sweep

### Objective
Identify the optimal combination of learning rate, hidden size, and self-supervised learning (SSL) pretraining strategy to maximize downstream multi-label diagnostic classification accuracy on the PTB-XL dataset.

### Parameters Under Evaluation
- **Learning Rates:** `[1e-3, 5e-4, 1e-4]`
- **LSTM Hidden State Sizes:** `[128, 256]`
- **SSL Pretraining Strategies:** `[None (Supervised Baseline), "mae", "contrastive"]`
- **Tuning Subset Size:** 3,000 records
- **Epochs:** 3 pretraining epochs, 5 fine-tuning/supervised epochs per trial

### Environment & Hardware
- **Hardware:** NVIDIA GeForce RTX 2050 (4GB) GPU
- **Frameworks:** PyTorch 2.5.1+cu121, MLflow 2.x
- **Tracking Database:** `sqlite:///mlflow.db`

### Tuning Progress & MLflow Dashboard
Trials are running as nested child runs grouped under a single parent run named `temporal_encoder_tuning_parent` in the `ECG_TemporalEncoder_Tuning` experiment. 

Epoch loss values, validation metrics (Subset Accuracy, Hamming Loss, Macro F1, Macro ROC-AUC), and serialized models are logged automatically.

To launch the dashboard and compare configurations:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## [2026-08-03] - Temporal Encoder Final Optimized Run

### Objective
Train a high-capacity BiLSTM model on the **full PTB-XL dataset** (17,418 training records) with optimized regularization, learning rate scheduling, and early stopping to maximize classification accuracy while preventing overfitting.

### Configuration
- **Model Architecture:** BiLSTM (input_size=12, hidden_size=256, num_layers=2)
- **SSL Pretraining:** Masked Autoencoder (MAE) pretraining (5 epochs)
- **Fine-Tuning/Supervised Epochs:** Up to 25 epochs
- **Regularization & Schedulers:**
  - LSTM Dropout: `0.4`
  - Fully Connected Head Dropout: `0.5`
  - Weight Decay: `1e-4`
  - LR Scheduler: `ReduceLROnPlateau` (factor 0.1, patience 3)
  - Early Stopping: `7 epochs` patience on validation loss
- **Resolution:** 100Hz (`lr`)
- **Batch Size:** 64
- **Experiment:** `ECG_TemporalEncoder_Optimized`
- **Run Name:** `ecg_blstm_final_optimized_run`
