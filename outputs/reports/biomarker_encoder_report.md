# ECG Biomarker Encoder Comprehensive Evaluation & Benchmarking Report

This report documents the architectural design, wave delineation correction, leakage-free retraining, representation quality analysis, and multi-label diagnostic evaluation of the **Biomarker Encoder Subsystem** on the full PTB-XL dataset (21,837 recordings).

---

## 1. Executive Summary

| Attribute | Specification / Finding |
|---|---|
| **Input Modality** | 24 domain-engineered electrophysiological biomarkers + 24 missingness indicators (48-D input vector) |
| **Delineation Method** | Continuous Wavelet Transform (CWT) multi-lead delineation (Lead II, V5, V1, I) |
| **Leakage Status** | **Zero Data Leakage** (Stratified patient-wise splitting; preprocessors fitted strictly on training subset) |
| **Extracted Dataset Size** | 21,808 successfully extracted 10-second recordings (99.87% extraction yield) |
| **Latent Representations** | 32-dimensional dense latent embeddings ($z \in \mathbb{R}^{32}$) with 0/32 collapsed dimensions |
| **Downstream Classification** | Multi-label Logistic Regression with class-weighted loss and validation threshold optimization |
| **Best Performing Model** | **Beta-VAE** (Macro F1 = **0.6347**, PR-AUC = **0.6918**) & **Attention MLP** (Macro F1 = **0.6332**, ROC-AUC = **0.8622**) |
| **Information Gain** | Learned embeddings outperform raw features by **+6.09% Macro F1** and **+13.48% PR-AUC** |

---

## 2. Delineation Method Audit & CWT Correction

### 2.1 The DWT Boundary Inflation Problem
Initial baseline extraction utilized the Discrete Wavelet Transform (DWT) delineator from NeuroKit2. Audit of the extracted feature distributions revealed systematic boundary overestimation:
- **DWT QRS Duration Overestimation**: The DWT delineator systematically placed QRS onsets too early and offsets too late, inflating median QRS duration to **169.62 ms** (83.3% abnormal QC warnings).
- **DWT PR Interval Compression**: Because QRS onset was placed early, the preceding PR interval was artificially compressed to a median of **98.31 ms** (51.2% abnormal QC warnings).

### 2.2 Continuous Wavelet Transform (CWT) Remediation
We replaced DWT with scale-adaptive Continuous Wavelet Transform (CWT) across all 21,808 records. CWT operates on continuous scale-space, providing high robustness against baseline wander and morphological noise.

| Metric / Feature | DWT (Old) Baseline | CWT (Corrected) Pipeline | Expected Physiological Normal | Clinical Verdict |
|---|---|---|---|---|
| **Median QRS Duration** | 169.62 ms | **105.79 ms** | 80.0 – 120.0 ms | **Normalized (Within Range)** |
| **Mean QRS Duration** | 171.71 ms | **106.89 ms** | 80.0 – 120.0 ms | **Normalized** |
| **QRS Duration Outliers** | 1.90% | **0.92%** | < 2.0% | **Low Outlier Rate** |
| **Median PR Interval** | 98.31 ms | **148.00 ms** | 120.0 – 200.0 ms | **Normalized (Within Range)** |
| **Mean PR Interval** | 106.44 ms | **146.58 ms** | 120.0 – 200.0 ms | **Normalized** |
| **QRS QC Warning Count** | 18,169 records | **604 records** | — | **-96.7% Warning Reduction** |
| **PR QC Warning Count** | 11,173 records | **2,244 records** | — | **-79.9% Warning Reduction** |

---

## 3. Leakage-Free Preprocessing Protocol

1. **Stratified Patient-Wise Partitioning**:
   - Split 21,808 records by `patient_id` into Train (70%), Validation (10%), and Test (20%).
   - Verified **0% patient overlap** across all partitions.
2. **Strict Preprocessor Fitting**:
   - `SimpleImputer(strategy='median')` and `StandardScaler()` were fitted **strictly on the training fold**.
   - Validation and test partitions were transformed using the frozen training parameters.
   - Missingness binary masks ($M \in \{0, 1\}^{24}$) were appended to preserve missingness topology, creating a 48-dimensional model input vector.

---

## 4. Comprehensive Model Benchmarks (Test Set)

We evaluated the three biomarker autoencoders against raw and preprocessed baseline features:

| Representation Source | Params | Recon MSE | Recon MAE | Macro F1 | Weighted F1 | ROC-AUC (Macro) | PR-AUC (Macro) |
|---|---|---|---|---|---|---|---|
| **Raw Features (CWT)** | — | — | — | 0.5738 | 0.6122 | 0.8040 | 0.5570 |
| **Preprocessed Features (CWT)** | — | — | — | 0.5885 | 0.6287 | 0.8158 | 0.5834 |
| **FT-Transformer Embedding** | 74,173 | **0.1286** | **0.2423** | 0.6192 | 0.6608 | 0.8575 | 0.6786 |
| **Attention MLP Embedding** | 205,789 | 0.1813 | 0.2950 | 0.6332 | 0.6737 | **0.8622** | 0.6876 |
| **Beta-VAE Embedding** | 108,829 | 0.4130 | 0.4654 | **0.6347** | **0.6767** | 0.8607 | **0.6918** |

---

## 5. Latent Space Representation Diagnostics

To ensure that the 32-dimensional latent vectors do not suffer from dimensional collapse or pathological multicollinearity, we performed full covariance and latent variance analysis:

| Model Architecture | Latent Dim | Collapsed Dims ($\sigma < 0.01$) | Total Latent Variance | Mean Abs Correlation | Highly Correlated Pairs ($|r| > 0.8$) |
|---|---|---|---|---|---|
| **Attention MLP** | 32 | **0 / 32** | **99.55** | 0.2452 | 1 |
| **Beta-VAE** | 32 | **0 / 32** | 3.93 | 0.3146 | 18 |
| **FT-Transformer** | 32 | **0 / 32** | 22.96 | 0.3069 | **0** |

### Key Takeaways:
- **Zero Dimension Collapse**: All 32 dimensions across all three architectures are active and participate in encoding.
- **Attention MLP**: Exhibits the highest latent variance and lowest inter-dimensional correlation, maximizing downstream expressive capacity.
- **FT-Transformer**: Achieves the lowest reconstruction error and strictly zero highly-correlated pairs, proving optimal for structural feature preservation.
- **Beta-VAE**: KL divergence regularization aligns the latent manifold into a continuous Gaussian space, achieving the highest overall downstream F1 and PR-AUC.

---

## 6. Per-Class Multi-Label Diagnostic Performance

Class-weighted Logistic Regression evaluated on the test set across 5 primary diagnostic superclasses:

| Model / Source | Metric | NORM | MI | STTC | CD | HYP | Macro Avg |
|---|---|---|---|---|---|---|---|
| **Attention MLP Embedding** | F1-Score | **0.8023** | 0.6281 | 0.6127 | **0.6140** | 0.5087 | **0.6332** |
| | ROC-AUC | **0.9023** | 0.8444 | 0.8699 | **0.8360** | **0.8585** | **0.8622** |
| | PR-AUC | **0.8730** | 0.6604 | 0.6512 | 0.6717 | 0.5818 | **0.6876** |
| **Beta-VAE Embedding** | F1-Score | 0.8112 | **0.6327** | **0.6213** | 0.6000 | 0.5081 | **0.6347** |
| | ROC-AUC | 0.9004 | **0.8531** | **0.8708** | 0.8345 | 0.8447 | 0.8607 |
| | PR-AUC | 0.8700 | **0.6829** | **0.6540** | **0.6833** | 0.5685 | **0.6918** |
| **FT-Transformer Embedding** | F1-Score | 0.8017 | 0.6226 | 0.5778 | 0.5802 | **0.5135** | 0.6192 |
| | ROC-AUC | 0.8999 | 0.8493 | 0.8548 | 0.8294 | 0.8543 | 0.8575 |
| | PR-AUC | 0.8717 | 0.6703 | 0.6199 | 0.6432 | **0.5878** | 0.6786 |
| **Preprocessed Baselines** | F1-Score | 0.7714 | 0.5536 | 0.5767 | 0.5545 | 0.4863 | 0.5885 |
| | ROC-AUC | 0.8686 | 0.7844 | 0.8421 | 0.7804 | 0.8034 | 0.8158 |
| | PR-AUC | 0.8300 | 0.5117 | 0.5701 | 0.5396 | 0.4658 | 0.5834 |

---

## 7. Recommendations for Downstream Fusion Engine

1. **Joint 64-D Latent Concatenation**:
   - Concatenate the 32-D latent vector from `Attention MLP` ($z_{\text{att}}$) and the 32-D latent vector from `Beta-VAE` ($z_{\text{vae}}$) to form a robust 64-D biomarker embedding:
     $$z_{\text{biomarker}} = [z_{\text{att}} \parallel z_{\text{vae}}] \in \mathbb{R}^{64}$$
   - This captures both the high-variance discriminative features of Attention MLP and the smooth continuous probability manifold of Beta-VAE.
2. **Multi-Modal Cross-Attention Fusion**:
   - Feed $z_{\text{biomarker}}$ alongside the Temporal Encoder embedding ($z_{\text{temp}} \in \mathbb{R}^{128}$) and Morphology Encoder embedding ($z_{\text{morph}} \in \mathbb{R}^{128}$) into the Adaptive Fusion Engine.
