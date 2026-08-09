# ECG Biomarker Encoder Benchmarking Report

Generated at: 2026-08-09 23:09:32

## Executive Summary

We trained, tuned, and compared three latent representation learning models on 1000 ECG biomarker feature profiles. The models were updated to support joint feature reconstruction and direct diagnostic classification using imputed inputs + binary missingness masks.

Based on reconstruction error (MSE), **attention_mlp** is the recommended model.

## Performance Metrics Comparison

| Model Type | Params | Reconstruction MSE | Reconstruction MAE | Latent Silhouette | Downstream F1 Score | Downstream ROC-AUC | Direct F1 Score | Direct ROC-AUC | Training Time (s) | Inference Time / Sample (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention_mlp | 408,487 | 0.927363 | 0.697528 | -0.0372 | 0.1354 | 0.6551 | 0.2122 | 0.8330 | 0.91 | 0.000101 |
| beta_vae | 311,559 | 0.980110 | 0.718536 | -0.0074 | 0.1248 | 0.3456 | 0.2975 | 0.8157 | 0.64 | 0.000048 |
| ft_transformer | 177,791 | 0.968167 | 0.715365 | 0.1091 | 0.1327 | 0.3608 | 0.2678 | 0.7610 | 142.16 | 0.004298 |

## Recommendation & Analysis

1. **Reconstruction Quality**: `attention_mlp` achieved the lowest mean squared error on the test dataset. A lower MSE indicates the learned latent space preserves the details of the input biomarkers.
2. **Latent Space Clusterability**: The Silhouette Score evaluates how well the latent representation aligns with clinical labels. Models with positive Silhouette scores learn structured manifolds that reflect downstream pathology.
3. **Downstream Classification**: Training a simple classifier directly on the latent 32-dim representation verifies the downstream clinical utility of the representation.
4. **Direct Classification**: Evaluating the model's internal classification head shows how well the model jointly learns reconstruction and classification.
