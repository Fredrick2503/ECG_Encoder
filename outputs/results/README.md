# Biomarker Encoder Results & Evaluation Artifacts

This directory contains standardized benchmark datasets, evaluation CSVs, and artifact references for the Biomarker Encoder module evaluated on the PTB-XL dataset (21,808 records).

---

## 1. Results Summary Table

| Representation Model / Baseline | Parameters | Recon MSE | Recon MAE | Macro F1 | Weighted F1 | ROC-AUC (Macro) | PR-AUC (Macro) |
|---|---|---|---|---|---|---|---|
| **Raw Features (CWT Baseline)** | — | — | — | 0.5738 | 0.6122 | 0.8040 | 0.5570 |
| **Preprocessed Baseline (Train-Fit)** | — | — | — | 0.5885 | 0.6287 | 0.8158 | 0.5834 |
| **FT-Transformer Autoencoder** | 74,173 | **0.1286** | **0.2423** | 0.6192 | 0.6608 | 0.8575 | 0.6786 |
| **Attention MLP Autoencoder** | 205,789 | 0.1813 | 0.2950 | 0.6332 | 0.6737 | **0.8622** | 0.6876 |
| **Beta-VAE Autoencoder** ($\beta=0.5$) | 108,829 | 0.4130 | 0.4654 | **0.6347** | **0.6767** | 0.8607 | **0.6918** |

---

## 2. Artifact Index

### 2.1 Tabular Metrics CSVs
- [`model_comparison_metrics_cwt.csv`](file:///c:/Users/Mallikarjuna/ECG_Encoder/outputs/results/model_comparison_metrics_cwt.csv): Complete multi-metric comparison table across models and baselines.
- [`per_class_metrics_cwt.csv`](file:///c:/Users/Mallikarjuna/ECG_Encoder/outputs/results/per_class_metrics_cwt.csv): Class-by-class breakdown (NORM, MI, STTC, CD, HYP) for F1, ROC-AUC, and PR-AUC.
- [`feature_qc_stats_cwt.csv`](file:///c:/Users/Mallikarjuna/ECG_Encoder/outputs/results/feature_qc_stats_cwt.csv): Statistical summary (min, max, median, mean, missingness, outlier rates) for all 24 extracted biomarkers.

### 2.2 Model Checkpoints
- [`biomarkers/attention_mlp_cwt.pt`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/attention_mlp_cwt.pt)
- [`biomarkers/beta_vae_cwt.pt`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/beta_vae_cwt.pt)
- [`biomarkers/ft_transformer_cwt.pt`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/ft_transformer_cwt.pt)

### 2.3 Diagnostic & Visualization Plots
- [`biomarkers/validation/qrs_trace_comparison_rec_1.png`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/validation/qrs_trace_comparison_rec_1.png): Visual overlay of DWT vs CWT wave delineation on raw ECG traces.
- [`biomarkers/validation/corrected_qrs_duration_distribution.png`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/validation/corrected_qrs_duration_distribution.png): Post-correction QRS duration distribution.
- [`biomarkers/validation/corrected_pr_interval_distribution.png`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/validation/corrected_pr_interval_distribution.png): Post-correction PR interval distribution.
- [`biomarkers/validation/attention_mlp_cwt_pca.png`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/validation/attention_mlp_cwt_pca.png): 2D PCA diagnostic class manifold for Attention MLP embeddings.
- [`biomarkers/validation/beta_vae_cwt_pca.png`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/validation/beta_vae_cwt_pca.png): 2D PCA diagnostic class manifold for Beta-VAE embeddings.
- [`biomarkers/validation/ft_transformer_cwt_pca.png`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/validation/ft_transformer_cwt_pca.png): 2D PCA diagnostic class manifold for FT-Transformer embeddings.
