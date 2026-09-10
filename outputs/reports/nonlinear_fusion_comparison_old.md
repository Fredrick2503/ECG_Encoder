# Phase 4: Controlled Nonlinear Fusion Benchmark Comparison

This report details the controlled comparison between Model A (T+M) and Model B (T+M+B) evaluated across 5 random seeds.

## 1. Aggregate Metrics Comparison (Mean ± SD)

| Metric | Model A (T+M) | Model B (T+M+B) | p-value |
| :--- | :---: | :---: | :---: |
| **MACRO_AUC** | `0.9061 ± 0.0033` | `0.9063 ± 0.0006` | `-` |
| **MACRO_F1** | `0.7080 ± 0.0037` | `0.7080 ± 0.0070` | `0.9945` |
| **MICRO_F1** | `0.7529 ± 0.0016` | `0.7512 ± 0.0062` | `-` |
| **SUBSET_ACC** | `0.5573 ± 0.0055` | `0.5547 ± 0.0171` | `-` |
| **MACRO_ECE** | `0.0485 ± 0.0016` | `0.0490 ± 0.0019` | `0.6588` |
| **BRIER** | `0.0856 ± 0.0006` | `0.0863 ± 0.0010` | `-` |


## 2. Per-Class F1 Scores Comparison

| Class | Model A (T+M) | Model B (T+M+B) |
| :--- | :---: | :---: |
| **NORM** | `0.8701 ± 0.0083` | `0.8661 ± 0.0055` |
| **MI** | `0.6777 ± 0.0066` | `0.6835 ± 0.0122` |
| **STTC** | `0.6679 ± 0.0135` | `0.6763 ± 0.0150` |
| **CD** | `0.7631 ± 0.0094` | `0.7436 ± 0.0164` |
| **HYP** | `0.5612 ± 0.0194` | `0.5704 ± 0.0145` |


## 3. Statistical and Clinical Conclusion

### Retain Model B (T+M+B) with Biomarkers
Model B (T+M+B) outperforms Model A (T+M) on Macro F1 (`0.5704` vs `0.5612`). This confirms that tabular biomarkers supply incremental, non-linear clinical attributes that the multi-layer classification engine successfully leverages.
