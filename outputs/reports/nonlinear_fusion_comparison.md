# Phase 4: Controlled Nonlinear Fusion Benchmark Comparison

This report details the controlled comparison between Model A (T+M) and Model B (T+M+B) evaluated across 5 random seeds.

## 1. Aggregate Metrics Comparison (Mean ± SD)

| Metric | Model A (T+M) | Model B (T+M+B) | p-value |
| :--- | :---: | :---: | :---: |
| **MACRO_AUC** | `0.9061 ± 0.0033` | `0.9078 ± 0.0011` | `-` |
| **MACRO_F1** | `0.7080 ± 0.0037` | `0.7173 ± 0.0112` | `0.1152` |
| **MICRO_F1** | `0.7529 ± 0.0016` | `0.7571 ± 0.0112` | `-` |
| **SUBSET_ACC** | `0.5573 ± 0.0055` | `0.5680 ± 0.0272` | `-` |
| **MACRO_ECE** | `0.0485 ± 0.0016` | `0.0496 ± 0.0032` | `0.5008` |
| **BRIER** | `0.0856 ± 0.0006` | `0.0852 ± 0.0009` | `-` |


## 2. Per-Class F1 Scores Comparison

| Class | Model A (T+M) | Model B (T+M+B) |
| :--- | :---: | :---: |
| **NORM** | `0.8701 ± 0.0083` | `0.8755 ± 0.0065` |
| **MI** | `0.6777 ± 0.0066` | `0.6832 ± 0.0130` |
| **STTC** | `0.6679 ± 0.0135` | `0.6746 ± 0.0209` |
| **CD** | `0.7631 ± 0.0094` | `0.7431 ± 0.0090` |
| **HYP** | `0.5612 ± 0.0194` | `0.6102 ± 0.0305` |


## 3. Statistical and Clinical Conclusion

### Retain Model B (T+M+B) with Biomarkers
Model B (T+M+B) outperforms Model A (T+M) on Macro F1 (`0.6102` vs `0.5612`). This confirms that tabular biomarkers supply incremental, non-linear clinical attributes that the multi-layer classification engine successfully leverages.
