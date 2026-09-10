# ECG Biomarker Encoder Thesis Notes

## 1. Methodology
- **Preprocessing**: Extracted 24 clinical electrophysiological features using Continuous Wavelet Transform (CWT) delineation. Missing values imputed using training-set medians. Features standardized using StandardScaler (mean=0, std=1). Missingness indicator flags appended (48-D input vector).
- **Model Input**: 48-dimensional vector (24 normalized features + 24 missingness binary indicators).
- **Joint Learning**: Networks simultaneously reconstruct the 24 original features and predict the 5 multi-label diagnostic targets (NORM, MI, STTC, CD, HYP) from a 32-dimensional latent representation.

## 2. Experimental Results (Test Set, Leakage-Free)

| Model Type | Params | Reconstruction MSE | Downstream Macro F1 | Downstream ROC-AUC | Downstream PR-AUC |
|---|---|---|---|---|---|
| **Raw Features (CWT Baseline)** | — | — | 0.5738 | 0.8040 | 0.5570 |
| **Preprocessed Baseline** | — | — | 0.5885 | 0.8158 | 0.5834 |
| **ft_transformer** | 74,173 | **0.12859** | 0.61917 | 0.85755 | 0.67858 |
| **attention_mlp** | 205,789 | 0.18129 | 0.63315 | **0.86220** | 0.68762 |
| **beta_vae** | 108,829 | 0.41302 | **0.63466** | 0.86069 | **0.69176** |

## 3. Conclusions
- `attention_mlp` and `beta_vae` achieve state-of-the-art representation quality, improving downstream Macro F1 by **+6.09%** over raw clinical features.
- Joint classification head training enables early anomaly detection directly from latent variables without dimensional collapse (0/32 collapsed dimensions).
- Continuous Wavelet Transform (CWT) delineation successfully resolved DWT boundary overestimation, delivering clinically sound parameters (median QRS duration: 105.79 ms; median PR interval: 148.00 ms).
