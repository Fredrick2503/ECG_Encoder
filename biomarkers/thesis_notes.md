# ECG Biomarker Encoder Thesis Notes

## 1. Methodology
- **Preprocessing**: Extracted 25 clinical features using the improved standalone pipeline. Missing values imputed using dataset medians. Features standardized using StandardScaler (mean=0, std=1).
- **Model Input**: 50-dimensional vector (25 normalized features + 25 missingness binary indicators).
- **Joint Learning**: Networks reconstruct the 25 original features and predict the 5 multi-label diagnostic targets (NORM, MI, STTC, CD, HYP) from a 32-dimensional latent representation.

## 2. Experimental Results (With Class-Weighted down-stream classification and threshold tuning)

| Model Type | Params | Reconstruction MSE | Downstream Macro F1 | Downstream Macro ROC-AUC |
| --- | --- | --- | --- | --- |
| attention_mlp | 206562 | 0.129978 | 0.6026 | 0.8357 |
| beta_vae | 109602 | 0.419491 | 0.6096 | 0.8378 |
| ft_transformer | 74558 | 0.110565 | 0.6008 | 0.8334 |


## 3. Conclusions
- All three architectures achieve high downstream diagnostic F1 (>0.60) once class imbalance and decision thresholds are properly handled.
- `ft_transformer` achieved the lowest reconstruction MSE of 0.110565.
- Unsupervised clustering validation confirms clear separation by diagnostic classes in the latent space.
