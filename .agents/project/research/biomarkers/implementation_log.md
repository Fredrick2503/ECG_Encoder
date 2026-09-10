# Biomarker Encoder Implementation Log

This document lists the code files, data artifacts, configuration parameters, and implementation details for the Biomarker Encoder module.

---

## 1. Source Files & Execution Pipeline

| File Path | Description | Key Classes / Functions |
|---|---|---|
| [`biomarkers/extract_features_cwt.py`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/extract_features_cwt.py) | Full dataset feature extraction using CWT | `clean_signal()`, `get_isoelectric_baseline()`, `calculate_axis()`, `calculate_qrs_t_angle()`, `extract_record_features()` |
| [`biomarkers/models.py`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/models.py) | PyTorch Neural Encoder Architectures | `ResidualBlock`, `AttentionMLPAutoencoder`, `BetaVAE`, `FeatureTokenizer`, `FTTransformerAutoencoder` |
| [`biomarkers/validation/retrain_corrected.py`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/validation/retrain_corrected.py) | Leakage-free training & evaluation script | `train_epoch()`, `evaluate()`, `train_model()`, `evaluate_representation()` |
| [`biomarkers/validation/run_audit.py`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/validation/run_audit.py) | Latent space diagnostics & QC auditor | Latent stats calculation, covariance analysis, silhouette scoring, PCA & t-SNE generation |
| [`biomarkers/evaluator.py`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/evaluator.py) | Downstream Logistic Regression evaluator | Multi-label threshold optimization, per-class F1/ROC/PR scoring |

---

## 2. Checkpoint & Preprocessing Artifacts

- **Attention MLP**: [`biomarkers/attention_mlp_cwt.pt`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/attention_mlp_cwt.pt) (205,789 parameters)
- **Beta-VAE**: [`biomarkers/beta_vae_cwt.pt`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/beta_vae_cwt.pt) (108,829 parameters, $\beta = 0.5$)
- **FT-Transformer**: [`biomarkers/ft_transformer_cwt.pt`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/ft_transformer_cwt.pt) (74,173 parameters, 2 layers, 4 heads)
- **Fitted Imputer**: [`biomarkers/imputer_cwt.pkl`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/imputer_cwt.pkl) (Median strategy on train split)
- **Fitted Scaler**: [`biomarkers/scaler_cwt.pkl`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/scaler_cwt.pkl) (StandardScaler on train split)

---

## 3. Training Hyperparameters

- **Input Dimension**: 48 (24 scaled features + 24 binary missing indicators)
- **Latent Dimension**: 32
- **Batch Size**: 128 (Train), 256 (Validation / Test)
- **Optimizer**: AdamW ($\text{lr} = 1\times 10^{-3}$, $\text{weight\_decay} = 1\times 10^{-4}$)
- **Learning Rate Scheduler**: `ReduceLROnPlateau(mode='min', factor=0.5, patience=5)`
- **Multi-Task Loss Balance**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + 1.0 \times \mathcal{L}_{\text{cls}} (+ 0.5 \times \mathcal{L}_{\text{KLD}} \text{ for VAE})$
- **Patience**: 15 epochs early stopping
