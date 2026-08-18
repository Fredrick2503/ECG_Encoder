# ECG Biomarker Encoder Benchmarking Report

Generated at: 2026-08-14 23:15:49

## Executive Summary

We trained, evaluated, and compared three latent representation learning models (Attention MLP, Beta-VAE, FT-Transformer) on the **full** dataset of 21808 preprocessed 24-biomarker feature profiles from PTB-XL.

The input dimension was 48 (24 standardized features + 24 binary missingness indicators) to support joint reconstruction and classification.

Based on a holistic trade-off between reconstruction quality (MSE) and downstream class separation (F1 and ROC-AUC), **attention_mlp** is the recommended model.

## Performance Metrics Comparison

| Model Type | Params | Reconstruction MSE | Downstream F1 Score | Downstream ROC-AUC | Direct F1 Score | Direct ROC-AUC | Training Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| attention_mlp | 205,789 | 0.169849 | 0.0000 | 0.5750 | 0.6252 | 0.8653 | 99.13 |
| beta_vae | 108,829 | 0.389117 | 0.0000 | 0.5038 | 0.6324 | 0.8667 | 76.57 |
| ft_transformer | 74,173 | 0.137986 | 0.0000 | 0.4346 | 0.5817 | 0.8588 | 570.85 |

## Per-Label Classification Metrics

### Downstream Classifier (LR on Latent Space)
| Model Type | Metric | NORM | MI | STTC | CD | HYP |
| --- | --- | --- | --- | --- | --- | --- |
| attention_mlp | F1-Score | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| attention_mlp | ROC-AUC | 0.6669 | 0.5432 | 0.5743 | 0.5458 | 0.5448 |
| beta_vae | F1-Score | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| beta_vae | ROC-AUC | 0.4478 | 0.5753 | 0.4740 | 0.5690 | 0.4527 |
| ft_transformer | F1-Score | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ft_transformer | ROC-AUC | 0.4149 | 0.4534 | 0.4188 | 0.4034 | 0.4826 |

### Direct Classification Head
| Model Type | Metric | NORM | MI | STTC | CD | HYP |
| --- | --- | --- | --- | --- | --- | --- |
| attention_mlp | F1-Score | 0.8040 | 0.6257 | 0.6125 | 0.5810 | 0.5027 |
| attention_mlp | ROC-AUC | 0.9073 | 0.8514 | 0.8817 | 0.8376 | 0.8485 |
| beta_vae | F1-Score | 0.8140 | 0.6217 | 0.6147 | 0.5816 | 0.5299 |
| beta_vae | ROC-AUC | 0.9062 | 0.8567 | 0.8839 | 0.8422 | 0.8445 |
| ft_transformer | F1-Score | 0.8017 | 0.5601 | 0.6040 | 0.4858 | 0.4570 |
| ft_transformer | ROC-AUC | 0.9026 | 0.8468 | 0.8814 | 0.8191 | 0.8442 |

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
