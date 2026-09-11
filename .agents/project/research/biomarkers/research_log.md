# Biomarker Encoder Research Log

This log documents research milestones, empirical findings, bug investigations, and architectural innovations for the Biomarker Encoder module.

---

### [2026-08-18] - Biomarker Representation Learning Formulation & Baseline Extraction
- **Topic:** Clinical Domain Feature Extraction & Autoencoder Modeling
- **Details:**
  - Designed the 24-feature clinical biomarker extraction pipeline across all 12 leads covering heart rate, conduction intervals (PR, QRS, QT, QTc), wave amplitudes (P, R, S, T), ST deviations (elevation, depression, count), electrical axes (QRS, T, spatial angle), and hypertrophy indices (Sokolow-Lyon).
  - Implemented multi-lead fallback hierarchy for P-wave delineation (Lead II -> V1 -> I) and R-peak alignment.
  - Implemented 3 autoencoder architectures in PyTorch: `AttentionMLPAutoencoder`, `BetaVAE`, and `FTTransformerAutoencoder`.
  - Formulated the joint multi-task loss function combining feature reconstruction MSE and 5-class multi-label BCEWithLogits loss.

---

### [2026-08-19] - Pipeline Validation & Detection of DWT Delineation Overestimation
- **Topic:** Quality Audit & Boundary Delineation Pathology Diagnosis
- **Details:**
  - Conducted a comprehensive quality audit of extracted features across 21,808 records.
  - **Identified DWT Measurement Inflation**: The NeuroKit2 Discrete Wavelet Transform (DWT) delineator was found to systematically place QRS onsets too early and offsets too late, causing median QRS duration to inflate to 169.62 ms (83.31% abnormal QC warnings) and compressing median PR interval to 98.31 ms (51.23% abnormal warnings).
  - **Identified Preprocessing Leakage**: Identified that `SimpleImputer` and `StandardScaler` had previously been fit on the full dataset before splitting.
  - Formulated remediation strategy: transition to Continuous Wavelet Transform (CWT) delineation and isolate imputer/scaler fitting to the training partition.

---

### [2026-08-20] - CWT Re-Extraction, Leakage-Free Retraining & Multi-Model Benchmarking
- **Topic:** Wavelet Scale-Space Correction & Downstream Benchmarking
- **Details:**
  - Re-extracted all 21,808 recordings using CWT delineation. Corrected median QRS duration to **105.79 ms** and median PR interval to **148.00 ms**, slashing QC warning flags by 96.7%.
  - Implemented `retrain_corrected.py` with strict patient-wise train/val/test partitioning (0% patient overlap) and training-set-only preprocessor fitting.
  - Retrained all three models (`Attention MLP`, `Beta-VAE`, and `FT-Transformer`) on CWT features.
  - Evaluated downstream multi-label diagnostic performance using Logistic Regression with validation-tuned decision thresholds.
  - Verified that learned 32-D latent embeddings dramatically outperform raw and preprocessed features (+6.09% Macro F1 gain, +13.48% PR-AUC gain).
  - Confirmed 0 collapsed dimensions and excellent latent space disentanglement across all models.
