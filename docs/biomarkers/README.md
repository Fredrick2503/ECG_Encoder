# Biomarker Encoder Subsystem

The Biomarker Encoder extracts 24 clinically validated electrophysiological biomarkers from 12-lead ECGs and projects them into regularized 32-dimensional latent representations for downstream pan-cardiac multi-label diagnosis.

---

## 1. Quickstart & Execution Workflow

### 1.1 Step 1: Run CWT Biomarker Extraction
Extracts 24 clinical features from 12-lead signals across PTB-XL using Continuous Wavelet Transform (CWT) delineation:
```bash
python biomarkers/extract_features_cwt.py
```
*Outputs: `biomarkers/ecg_biomarkers_full_cwt.csv`, `biomarkers/qc_logs_cwt.csv`, `biomarkers/full_extraction_report_cwt.md`*

### 1.2 Step 2: Leakage-Free Retraining & Checkpointing
Fits imputer and scaler strictly on the training partition and trains the three autoencoder architectures (`Attention MLP`, `Beta-VAE`, and `FT-Transformer`):
```bash
python biomarkers/validation/retrain_corrected.py
```
*Outputs: `biomarkers/*_cwt.pt`, `biomarkers/imputer_cwt.pkl`, `biomarkers/scaler_cwt.pkl`*

### 1.3 Step 3: Comprehensive Evaluation & Diagnostics
Generates downstream multi-label benchmark metrics, per-class evaluations, latent space PCA/t-SNE plots, and latent dimension correlation audits:
```bash
python biomarkers/validation/run_audit.py
```

---

## 2. Model Checkpoints & Preprocessing Artifacts

| Artifact | Path | Size | Description |
|---|---|---|---|
| **Attention MLP Model** | [`biomarkers/attention_mlp_cwt.pt`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/attention_mlp_cwt.pt) | 834 KB | 205k parameter self-attention autoencoder |
| **Beta-VAE Model** | [`biomarkers/beta_vae_cwt.pt`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/beta_vae_cwt.pt) | 445 KB | 108k parameter variational autoencoder ($\beta=0.5$) |
| **FT-Transformer Model** | [`biomarkers/ft_transformer_cwt.pt`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/ft_transformer_cwt.pt) | 311 KB | 74k parameter Feature-Tokenizer Transformer |
| **Imputer** | [`biomarkers/imputer_cwt.pkl`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/imputer_cwt.pkl) | 602 B | Training-set median imputer |
| **Scaler** | [`biomarkers/scaler_cwt.pkl`](file:///c:/Users/Mallikarjuna/ECG_Encoder/biomarkers/scaler_cwt.pkl) | 992 B | Training-set z-score standardizer |

---

## 3. Verified Performance Summary (Test Set)

| Representation Source | Macro F1 | Weighted F1 | ROC-AUC (Macro) | PR-AUC (Macro) | Recon MSE | Collapsed Dims |
|---|---|---|---|---|---|---|
| **Raw Features (CWT)** | 0.5738 | 0.6122 | 0.8040 | 0.5570 | — | — |
| **Preprocessed Features (CWT)** | 0.5885 | 0.6287 | 0.8158 | 0.5834 | — | — |
| **FT-Transformer Embedding** | 0.6192 | 0.6608 | 0.8575 | 0.6786 | **0.1286** | **0 / 32** |
| **Attention MLP Embedding** | 0.6332 | 0.6737 | **0.8622** | 0.6876 | 0.1813 | **0 / 32** |
| **Beta-VAE Embedding** | **0.6347** | **0.6767** | 0.8607 | **0.6918** | 0.4130 | **0 / 32** |

---

## 4. How to Extract Embeddings in Python

```python
import torch
import pickle
import numpy as np
import pandas as pd
from biomarkers.models import AttentionMLPAutoencoder, BetaVAE

# 1. Load preprocessors
with open("biomarkers/imputer_cwt.pkl", "rb") as f:
    imputer = pickle.load(f)
with open("biomarkers/scaler_cwt.pkl", "rb") as f:
    scaler = pickle.load(f)

# 2. Preprocess 24 raw clinical features
features_raw = np.random.randn(1, 24)  # (1, 24)
missing_mask = np.isnan(features_raw).astype(np.float32)
features_imputed = imputer.transform(features_raw)
features_scaled = scaler.transform(features_imputed)
model_input = torch.tensor(np.hstack([features_scaled, missing_mask]), dtype=torch.float32)

# 3. Load model and extract 32-D latent vector
model = AttentionMLPAutoencoder(input_dim=48, latent_dim=32)
model.load_state_dict(torch.load("biomarkers/attention_mlp_cwt.pt", map_location="cpu"))
model.eval()

with torch.no_grad():
    latent_32d = model.encode(model_input).numpy()
print("Extracted Biomarker Representation Shape:", latent_32d.shape) # (1, 32)
```
