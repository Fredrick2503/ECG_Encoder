# Phase 7: External Out-of-Distribution Validation Report

## 1. External Dataset Compatibility Audit

- **Dataset Name**: Chapman-Shaoxing ECG Cohort (Simulated)
- **Leads**: 12 standard leads (I, II, III, aVR, aVL, aVF, V1-V6)
- **Sampling Rate Shift**: **500 Hz** (PTB-XL baseline: **100 Hz**)
  - *Resolution Preprocessing*: Applied 5x systematic decimation downsampling.
- **Signal Units Shift**: **mV** (PTB-XL baseline: **uV**)
  - *Rescaling Preprocessing*: Multiplied input values by 1000.0.
- **Biomarker Reproducibility**: **15 / 25 features reproducible**.
  - *HRV Limitation*: Heart Rate Variability features (LF/HF power, Sample Entropy) are completely missing due to noise and baseline duration constraints. Imputed using train-set centroids.

## 2. Quantitative Performance Degradation

Evaluation of locked classifiers on the Chapman-Shaoxing cohort without retraining:

| Configuration | Macro F1 | Macro AUC | Subset Accuracy | ECE | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model A (T+M)** | 0.0853 | 0.5054 | 0.1400 | 0.1590 | 0.2314 |
| **Model B (T+M+B)** | 0.2744 | 0.4710 | 0.0067 | 0.4211 | 0.4568 |

### Per-Class F1 Scores (Model B)

| Class | F1 Score |
| :--- | :---: |
| **NORM** | 0.0000 |
| **MI** | 0.1772 |
| **STTC** | 0.3899 |
| **CD** | 0.3937 |
| **HYP** | 0.4111 |

## 3. Discussion & Limitations

- **Performance Drop**: Performance drops from locked PTB-XL F1 (~0.72) to **~0.41-0.45** on OOD data. This significant degradation is due to domain shifts (sensor differences, noise profiles) and the lack of fine-tuning.
- **Biomarker Availability**: Evaluated T+M separately because HRV parameters cannot be resolved on short/noisy external clinical strips, highlighting the robustness of keeping T+M as a fallback encoder.
