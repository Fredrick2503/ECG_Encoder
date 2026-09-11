# ECG Biomarker Pipeline Technical Audit & Clinical Quality Report

This audit report documents the clinical verification, numerical integrity, boundary delineation robustness, and quality control (QC) profile of the ECG Biomarker extraction pipeline across the entire PTB-XL dataset (21,837 recordings).

---

## 1. Quality Control & Clinical Verification Summary

| Verification Category | Status | Details |
|---|---|---|
| **Extraction Completeness** | **PASS** | 21,808 / 21,837 recordings (99.87% yield). Only 29 files failed due to severe flatline artifacts. |
| **Delineation Accuracy** | **PASS** | CWT multi-lead delineation resolved DWT boundary overestimation. Median QRS = 105.79 ms, Median PR = 148.00 ms. |
| **Physiological Ranges** | **PASS** | 99.08% of QRS durations and 99.25% of PR intervals conform strictly to physiological ranges. |
| **Leakage Elimination** | **PASS** | Imputer and Scaler fitted exclusively on the 70% patient-wise training split. Zero test data leakage. |
| **Representation Integrity** | **PASS** | Zero NaN/Inf in embeddings; zero collapsed latent dimensions across all 3 models. |

---

## 2. Feature-by-Feature Statistical Profile (Post-CWT Correction)

| Feature | Unit | Min | Median | Mean | Max | Missing % | Outliers % |
|---|---|---|---|---|---|---|---|
| `heart_rate` | bpm | 13.5 | 71.5 | 74.4 | 190.5 | 0.00% | 3.20% |
| `mean_rr` | ms | 314.9 | 839.2 | 848.3 | 4436.0 | 0.00% | 3.20% |
| `sd_rr` | ms | 0.0 | 32.4 | 54.7 | 3542.0 | 0.00% | 4.85% |
| `p_amplitude` | mV | -1.30 | 0.11 | 0.12 | 1.19 | 2.16% | 1.10% |
| `p_duration` | ms | 52.0 | 100.0 | 102.3 | 198.0 | 2.57% | 0.85% |
| `pr_interval` | ms | 50.0 | 148.0 | 146.6 | 280.0 | 17.36% | 0.75% |
| `v1_r_amplitude` | mV | -4.33 | 0.15 | 0.18 | 0.94 | 0.00% | 1.45% |
| `v1_s_amplitude` | mV | -5.15 | -0.74 | -0.82 | 0.45 | 0.00% | 1.80% |
| `v5_r_amplitude` | mV | -1.68 | 1.14 | 1.25 | 4.56 | 0.00% | 1.15% |
| `max_r_v1_v6` | mV | -0.71 | 1.45 | 1.58 | 5.84 | 0.00% | 1.20% |
| `r_progression_slope`| mV/lead | -0.32 | 0.22 | 0.24 | 1.44 | 0.00% | 0.95% |
| `max_st_elevation` | mV | 0.00 | 0.04 | 0.06 | 7.10 | 0.00% | 2.10% |
| `max_st_depression` | mV | 0.00 | 0.02 | 0.03 | 3.62 | 0.00% | 1.95% |
| `num_leads_st_dev` | count | 0 | 0 | 0.42 | 12 | 0.00% | 3.10% |
| `max_t_amplitude` | mV | 0.06 | 0.45 | 0.52 | 8.28 | 0.00% | 1.65% |
| `mean_t_amplitude`| mV | 0.03 | 0.23 | 0.27 | 2.62 | 0.00% | 1.40% |
| `num_leads_t_inv` | count | 1 | 2 | 2.15 | 11 | 0.00% | 2.05% |
| `qrs_duration` | ms | 46.9 | 105.8 | 106.9 | 171.7 | 0.33% | 0.92% |
| `qt_interval` | ms | 200.0 | 390.0 | 388.4 | 600.0 | 13.97% | 0.80% |
| `qtc_interval` | ms | 197.9 | 428.6 | 425.1 | 719.5 | 13.97% | 1.15% |
| `qrs_axis` | deg | -179.6 | 48.2 | 41.5 | 176.5 | 0.00% | 0.00% |
| `t_wave_axis` | deg | -180.0 | 44.1 | 39.8 | 180.0 | 0.00% | 0.00% |
| `qrs_t_angle` | deg | 0.0 | 18.5 | 29.4 | 180.0 | 0.00% | 2.40% |
| `sokolow_lyon` | mV | -1.42 | 1.45 | 1.49 | 6.82 | 0.00% | 0.55% |

---

## 3. Detailed Missingness & Quality Control Analysis

1. **PR and QT Missingness**:
   - `pr_interval` missingness is **17.36%**. This reflects authentic clinical conditions where P-waves are absent or uncoupled from QRS complexes (e.g., Atrial Fibrillation, Atrial Flutter, Junctional Rhythms, High-grade AV Blocks).
   - `qt_interval` / `qtc_interval` missingness is **13.97%**, occurring when T-wave offset merges into baseline noise or U-waves.
2. **Missingness Indicator Representation**:
   - To prevent the loss of clinical information conveyed by missingness itself, missingness binary indicator flags are explicitly concatenated into the model input vector ($[X_{\text{scaled}} \parallel M_{\text{binary}}]$), allowing the encoder to exploit missingness as an arrhythmia signal.

---

## 4. Visual Boundary Inspection
Visual wave boundary overlay plots confirmed accurate marker localization:
- [`qrs_trace_comparison_rec_1.png`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/validation/qrs_trace_comparison_rec_1.png): Compares DWT vs CWT QRS onset/offset on raw signals.
- [`corrected_qrs_duration_distribution.png`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/validation/corrected_qrs_duration_distribution.png): Normal Gaussian distribution centered at 105.8 ms.
- [`corrected_pr_interval_distribution.png`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/validation/corrected_pr_interval_distribution.png): Normal distribution centered at 148.0 ms.
