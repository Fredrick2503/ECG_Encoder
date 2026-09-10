# Implementation Plan - QRS/PR Delineation Correction & Re-Evaluation

This plan outlines the steps to replace the problematic DWT QRS delineation with the Continuous Wavelet Transform (CWT) method, re-extract QRS/PR biomarkers, retrain the models, and perform an Old vs Corrected comparison.

## User Review Required

> [!WARNING]
> - Continuous Wavelet Delineation (CWT) has a **high missingness rate (~36% for QRS, ~11% for PR)** compared to DWT (<1%).
> - Replacing DWT with CWT will require the imputer to fill in missing values using the median for these records.
> - We will evaluate whether this clinical correctness improvement helps, hurts, or has negligible impact on ML classification performance.
> - All corrected checkpoints, preprocessing files, and reports will be saved separately to preserve baseline results.

## Proposed Changes

### 1. Corrected Extraction
#### [NEW] [correct_extraction.py](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/validation/correct_extraction.py)
This script will:
- Load the raw PTB-XL ECG signals.
- Re-run Lead II delineation using CWT to obtain corrected `qrs_duration` and `pr_interval` values.
- Retain other 22 features from the original extraction.
- Save the new dataset to `biomarkers/ecg_biomarkers_full_corrected.csv`.
- Save distribution plots and statistics comparison.

### 2. Preprocessing & Split
- Perform the same patient-wise split.
- Fit imputer and scaler ONLY on corrected training data.
- Save preprocessing objects as `imputer_corrected.pkl` and `scaler_corrected.pkl`.

### 3. Model Retraining
- Retrain Attention MLP, Beta-VAE, and FT-Transformer models.
- Save model weights as `attention_mlp_corrected.pt`, `beta_vae_corrected.pt`, and `ft_transformer_corrected.pt`.

### 4. Downstream Evaluation & Reporting
- Evaluate downstream Logistic Regression models with optimized thresholds.
- Compare metrics of OLD (DWT) vs CORRECTED (CWT) pipelines.
- Generate report: `biomarkers/biomarker_encoder_corrected_validation_report.md`.

## Verification Plan

### Automated Tests
Run:
```powershell
python biomarkers/validation/correct_extraction.py
python biomarkers/validation/retrain_corrected.py
```
Check that output files are correctly created:
- `biomarkers/ecg_biomarkers_full_corrected.csv`
- `biomarkers/biomarker_encoder_corrected_validation_report.md`
- `biomarkers/validation/corrected_*_pca.png`
- Corrected models: `*_corrected.pt`
