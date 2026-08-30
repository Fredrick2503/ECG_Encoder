# Final Comprehensive Validation & Auditing Report (Age Group 18-30)

## 1. P0: Test-Set Leakage & Model Freeze Audit
- **Patient ID Overlaps**:
  - Train vs Validation: **0** patients overlapping.
  - Train vs Test:       **0** patients overlapping.
  - Validation vs Test:  **0** patients overlapping.
  *Verification Verdict*: **SUCCESS** (Strict patient-level separation is preserved).

- **Model Weights Checksums (MD5)**:
  - `classification_mlp_age_18_30.pt`: `3c18f892141268d422df3681b36d2c57`
  - `classification_mlp_age_18_30_tm.pt`: `01b57d9ad9d8c456f3a2c8e25bb7f8fb`

## 2. P0: Statistical Significance and Multi-Seed Comparison
- **Wilcoxon Signed-Rank Test p-value**: **`0.052020`**
- **Permutation Test p-value**: **`0.060000`**

### Model Performance Panel

| Model / Metric | Macro F1 | Macro AUC | Subset Accuracy | Macro ECE | Brier Score |
| --- | :---: | :---: | :---: | :---: | :---: |
| **Model A (T+M)** | 0.5953 | nan | 0.7727 | 0.3093 | 0.1322 |
| **Model B (T+M+B)** | 0.7953 | nan | 0.8182 | 0.3692 | 0.1695 |

## 3. P1: Biomarker Contribution & Correlation Analysis
- **QRS Amplitude Spearman Correlation with Hypertrophy Predictions**: $r = 0.0627$ (p-value: `0.7817`)

## 4. P1: Robustness Uncertainty & Lead Failure Mode Analysis
- **Clean Baseline F1:** `0.7953`
- **High-Frequency Noise F1:** `0.7953`
- **Chest-Leads Masking F1:** `0.7953`
