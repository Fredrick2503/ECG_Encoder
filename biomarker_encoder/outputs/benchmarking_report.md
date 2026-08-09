# ECG Biomarker Encoder Benchmarking Report

Generated at: 2026-08-09 23:00:43

## Executive Summary

We trained, tuned, and compared three latent representation learning models on 1000 ECG biomarker feature profiles. The models were updated to support joint feature reconstruction and direct diagnostic classification using imputed inputs + binary missingness masks.

Based on reconstruction error (MSE), **attention_mlp** is the recommended model.

## Performance Metrics Comparison

| Model Type | Params | Reconstruction MSE | Reconstruction MAE | Latent Silhouette | Downstream F1 Score | Downstream ROC-AUC | Direct F1 Score | Direct ROC-AUC | Training Time (s) | Inference Time / Sample (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention_mlp | 408,487 | 0.914655 | 0.690287 | -0.0711 | 0.1528 | 0.6638 | 0.3211 | 0.8331 | 1.41 | 0.000291 |
| beta_vae | 311,559 | 0.947364 | 0.704219 | 0.0557 | 0.1668 | 0.6489 | 0.3431 | 0.8324 | 2.01 | 0.000308 |
| ft_transformer | 177,791 | 0.966264 | 0.709030 | -0.0836 | 0.1111 | 0.4473 | 0.1822 | 0.7735 | 168.71 | 0.003501 |

## Recommendation & Analysis

1. **Reconstruction Quality**: `attention_mlp` achieved the lowest mean squared error on the test dataset. A lower MSE indicates the learned latent space preserves the details of the input biomarkers.
2. **Latent Space Clusterability**: The Silhouette Score evaluates how well the latent representation aligns with clinical labels. Models with positive Silhouette scores learn structured manifolds that reflect downstream pathology.
3. **Downstream Classification**: Training a simple classifier directly on the latent 32-dim representation verifies the downstream clinical utility of the representation.
4. **Direct Classification**: Evaluating the model's internal classification head shows how well the model jointly learns reconstruction and classification.
