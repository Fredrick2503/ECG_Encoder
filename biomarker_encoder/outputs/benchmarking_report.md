# ECG Biomarker Encoder Benchmarking Report

Generated at: 2026-08-09 22:09:06

## Executive Summary

We trained, tuned, and compared three latent representation learning models on 100 ECG biomarker feature profiles. The models were updated to support joint feature reconstruction and direct diagnostic classification using imputed inputs + binary missingness masks.

Based on reconstruction error (MSE), **ft_transformer** is the recommended model.

## Performance Metrics Comparison

| Model Type | Params | Reconstruction MSE | Reconstruction MAE | Latent Silhouette | Downstream F1 Score | Downstream ROC-AUC | Direct F1 Score | Direct ROC-AUC | Training Time (s) | Inference Time / Sample (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention_mlp | 214,464 | 0.779922 | 0.656600 | 0.1863 | 0.1846 | nan | 0.2000 | nan | 1.28 | 0.000232 |
| beta_vae | 71,504 | 0.824449 | 0.666312 | 0.4596 | 0.1926 | nan | 0.1920 | nan | 1.33 | 0.000037 |
| ft_transformer | 124,516 | 0.735105 | 0.628306 | 0.4224 | 0.1926 | nan | 0.1920 | nan | 7.08 | 0.000476 |

## Recommendation & Analysis

1. **Reconstruction Quality**: `ft_transformer` achieved the lowest mean squared error on the test dataset. A lower MSE indicates the learned latent space preserves the details of the input biomarkers.
2. **Latent Space Clusterability**: The Silhouette Score evaluates how well the latent representation aligns with clinical labels. Models with positive Silhouette scores learn structured manifolds that reflect downstream pathology.
3. **Downstream Classification**: Training a simple classifier directly on the latent 32-dim representation verifies the downstream clinical utility of the representation.
4. **Direct Classification**: Evaluating the model's internal classification head shows how well the model jointly learns reconstruction and classification.
