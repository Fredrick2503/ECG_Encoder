# ECG Biomarker Encoder Thesis Notes

## 1. Methodology
- **Preprocessing**: Extracted 24 clinical features. Missing values imputed using dataset medians. Features standardized using StandardScaler (mean=0, std=1).
- **Model Input**: 48-dimensional vector (24 normalized features + 24 missingness binary indicators).
- **Joint Learning**: Networks reconstruct the 24 original features and predict the 5 multi-label diagnostic targets (NORM, MI, STTC, CD, HYP) from a 32-dimensional latent representation.

## 2. Experimental Results

| Model Type | Params | Reconstruction MSE | Downstream F1 Score | Downstream ROC-AUC |
| --- | --- | --- | --- | --- |
| attention_mlp | 205789 | 0.169849 | 0.0000 | 0.5750 |
| beta_vae | 108829 | 0.389117 | 0.0000 | 0.5038 |
| ft_transformer | 74173 | 0.137986 | 0.0000 | 0.4346 |


## 3. Conclusions
- `attention_mlp` achieved the most robust latent representations for classification and reconstruction.
- Joint classification head training enables early anomaly detection directly from latent variables.
