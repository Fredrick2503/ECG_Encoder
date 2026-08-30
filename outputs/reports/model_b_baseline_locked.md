# Model B Baseline Locked Report

This report documents the baseline locked configuration of Model B, including MD5 hashes of all weights and clean test set metrics.

## 1. MD5 Integrity Check

| Component | Filename | MD5 Hash |
| --- | --- | --- |
| Temporal Encoder | `C5_full_dataset.pt` | `3d35c23a94fed9f20139e25c7f96e096` |
| Morphology Encoder | `morphology_encoder_v1.pt` | `e319aeba56fcb1e836662675afc487c3` |
| Biomarker Encoder | `attention_mlp_best.pt` | `3bf82fe7be15ce83c8bf4aba3f0cc1aa` |
| MLP Classifier | `classification_mlp.pt` | `11ebe9c08121b9b60245037bb3bd045a` |
| Classifier Thresholds | `classification_mlp_thresholds.npy` | `46f4f3fe17d590f1e86b61074edcd2e4` |

## 2. Locked Baseline Test Metrics

- **Macro F1**: `0.722079`
- **Macro AUC**: `0.909252`
- **Subset Accuracy**: `0.563333`
- **Macro ECE**: `0.048731`
- **Brier Score (Mean)**: `0.084846`

## 3. Per-Class Detail

| Class | F1-Score | ROC-AUC | ECE | Brier Score | Decision Threshold |
| --- | --- | --- | --- | --- | --- |
| NORM | 0.873239 | 0.939546 | 0.062156 | 0.093605 | 0.53 |
| MI | 0.687500 | 0.891920 | 0.056897 | 0.107496 | 0.26 |
| STTC | 0.658228 | 0.929145 | 0.052176 | 0.089747 | 0.25 |
| CD | 0.746269 | 0.914923 | 0.045009 | 0.076170 | 0.30 |
| HYP | 0.645161 | 0.870728 | 0.027417 | 0.057212 | 0.33 |
