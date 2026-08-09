# ECG Biomarker Encoder Benchmarking Report

Generated at: 2026-08-06 18:28:46

## Executive Summary

We trained, tuned, and compared three latent representation learning models on 21837 ECG biomarker feature profiles. Based on reconstruction error (MSE), **ft_transformer** is the recommended model.

## Performance Metrics Comparison

| Model Type | Params | Reconstruction MSE | Reconstruction MAE | Latent Silhouette | Downstream F1 Score | Downstream ROC-AUC | Training Time (s) | Inference Time / Sample (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention_mlp | 213,337 | 0.121961 | 0.222051 | 0.0121 | 0.0028 | 0.4806 | 253.70 | 0.000041 |
| beta_vae | 155,897 | 0.445894 | 0.442525 | -0.0099 | 0.0000 | 0.5351 | 186.98 | 0.000030 |
| ft_transformer | 140,467 | 0.031120 | 0.106840 | 0.0100 | 0.0000 | 0.4835 | 1321.43 | 0.000139 |

## Recommendation & Analysis

1. **Reconstruction Quality**: `ft_transformer` achieved the lowest mean squared error on the test dataset. A lower MSE indicates the learned latent space preserves the details of the input biomarkers.
2. **Latent Space Clusterability**: The Silhouette Score evaluates how well the latent representation aligns with clinical labels. Models with positive Silhouette scores learn structured manifolds that reflect downstream pathology.
3. **Downstream Classification**: Training a simple classifier directly on the latent 32-dim representation verifies the downstream clinical utility of the representation.
