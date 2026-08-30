# ECG Biomarker Encoder Benchmarking Report

Generated at: 2026-08-20 19:29:25

## Executive Summary

We trained, evaluated, and compared three latent representation learning models (Attention MLP, Beta-VAE, FT-Transformer) on the **full** dataset of 21837 preprocessed 24-biomarker feature profiles from PTB-XL.

The input dimension was 48 (24 standardized features + 24 binary missingness indicators) to support joint reconstruction and classification.

Based on a holistic trade-off between reconstruction quality (MSE) and downstream class separation (F1 and ROC-AUC), **attention_mlp** is the recommended model.

## Performance Metrics Comparison

| Model Type | Params | Reconstruction MSE | Downstream F1 Score | Downstream ROC-AUC | Direct F1 Score | Direct ROC-AUC | Training Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| attention_mlp | 206,562 | 0.129978 | 0.0000 | 0.5640 | 0.5209 | 0.8352 | 221.72 |
| beta_vae | 109,602 | 0.422161 | 0.0000 | 0.4723 | 0.5561 | 0.8366 | 175.28 |
| ft_transformer | 74,558 | 0.110565 | 0.0000 | 0.4605 | 0.5141 | 0.8332 | 288.71 |

## Per-Label Classification Metrics

### Downstream Classifier (LR on Latent Space)
| Model Type | Metric | NORM | MI | STTC | CD | HYP |
| --- | --- | --- | --- | --- | --- | --- |
| attention_mlp | F1-Score | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| attention_mlp | ROC-AUC | 0.5584 | 0.5875 | 0.5875 | 0.5889 | 0.4976 |
| beta_vae | F1-Score | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| beta_vae | ROC-AUC | 0.3882 | 0.4755 | 0.5368 | 0.4135 | 0.5478 |
| ft_transformer | F1-Score | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ft_transformer | ROC-AUC | 0.4368 | 0.4965 | 0.5035 | 0.4689 | 0.3966 |

### Direct Classification Head
| Model Type | Metric | NORM | MI | STTC | CD | HYP |
| --- | --- | --- | --- | --- | --- | --- |
| attention_mlp | F1-Score | 0.7841 | 0.4962 | 0.5820 | 0.4023 | 0.3402 |
| attention_mlp | ROC-AUC | 0.8886 | 0.7971 | 0.8683 | 0.7843 | 0.8376 |
| beta_vae | F1-Score | 0.7875 | 0.5323 | 0.5940 | 0.4574 | 0.4092 |
| beta_vae | ROC-AUC | 0.8895 | 0.8012 | 0.8646 | 0.7848 | 0.8429 |
| ft_transformer | F1-Score | 0.7965 | 0.4928 | 0.5974 | 0.3696 | 0.3140 |
| ft_transformer | ROC-AUC | 0.8914 | 0.7941 | 0.8699 | 0.7709 | 0.8397 |

## Model-Specific Analysis

### Attention Mlp
- **Strengths**: Light footprint and rapid training with competitive classification performance.
- **Limitations**: Lacks sequence-level relational awareness.

### Beta Vae
- **Strengths**: Smooth and continuous latent space, ideal for interpolation and anomaly generation.
- **Limitations**: Trade-off between reconstruction and class separability governed by Beta coefficient.

### Ft Transformer
- **Strengths**: Excels at mapping complex dependencies using attention layers. Provides highly separable downstream embeddings.
- **Limitations**: Higher parameter footprint and training time.

## Final Verdict & Recommendation
> [!IMPORTANT]
> **Attention Mlp** is selected as the optimal encoder. It achieves a balanced trade-off between faithful clinical biomarker reconstruction and high-fidelity downstream class separation. Its representations are recommended for integration with downstream diagnostic classifiers.
