# Biomarker Encoder Low-Level Design (LLD)

## 1. Clinical Biomarker Specification (24 Features)

The feature extraction engine extracts 24 clinical features from 12-lead ECG signals:

| Index | Feature Name | Clinical Description | Leads / Source | Expected Physiological Range |
|---|---|---|---|---|
| 1 | `heart_rate` | Ventricular rate in beats per minute | Lead II R-R intervals | 50.0 – 100.0 bpm |
| 2 | `mean_rr` | Mean R-R interval duration | Lead II | 600.0 – 1200.0 ms |
| 3 | `sd_rr` | Standard deviation of R-R intervals (HRV metric) | Lead II | 10.0 – 150.0 ms |
| 4 | `p_amplitude` | P-wave peak amplitude relative to isoelectric baseline | Lead II (fallback: V1, I) | 0.05 – 0.25 mV |
| 5 | `p_duration` | P-wave onset to offset interval | Lead II | 60.0 – 120.0 ms |
| 6 | `pr_interval` | P-wave onset to QRS onset interval | Lead II | 120.0 – 200.0 ms |
| 7 | `v1_r_amplitude` | R-wave peak amplitude in Lead V1 | Lead V1 | 0.0 – 0.7 mV |
| 8 | `v1_s_amplitude` | S-wave nadir amplitude in Lead V1 | Lead V1 | -0.3 – -2.5 mV |
| 9 | `v5_r_amplitude` | R-wave peak amplitude in Lead V5 | Lead V5 | 0.6 – 2.5 mV |
| 10 | `max_r_v1_v6` | Maximum R-wave amplitude across precordial leads | Leads V1–V6 | 0.8 – 3.0 mV |
| 11 | `r_progression_slope` | Linear slope of R-wave amplitude across V1–V5 | Leads V1–V5 | Positive (> 0.05) |
| 12 | `max_st_elevation` | Maximum ST-segment elevation at J-point + 60ms | Leads I, II, III, aVF, V1–V6 | < 0.1 mV (limb), < 0.2 mV (precordial) |
| 13 | `max_st_depression` | Maximum ST-segment depression at J-point + 60ms | All 12 leads | < 0.05 mV |
| 14 | `num_leads_st_deviation` | Total number of leads with significant ST shift | All 12 leads | 0 leads |
| 15 | `max_t_amplitude` | Maximum T-wave peak amplitude | All 12 leads | 0.1 – 1.0 mV |
| 16 | `mean_t_amplitude` | Average T-wave amplitude | All 12 leads | 0.1 – 0.5 mV |
| 17 | `num_leads_t_inversion` | Number of leads exhibiting negative T-waves | Leads I, II, V3–V6 | 0 – 1 leads (aVR normal) |
| 18 | `qrs_duration` | QRS complex onset to offset interval | Lead II (CWT) | 80.0 – 120.0 ms |
| 19 | `qt_interval` | QRS onset to T-wave offset interval | Lead II | 350.0 – 450.0 ms |
| 20 | `qtc_interval` | Bazett-corrected QT interval: $QT / \sqrt{RR}$ | Lead II | 360.0 – 440.0 ms (M), 460.0 ms (F) |
| 21 | `qrs_axis` | Frontal electrical QRS axis | Leads I & aVF | -30° to +90° |
| 22 | `t_wave_axis` | Frontal T-wave electrical axis | Leads I & aVF | 0° to +75° |
| 23 | `qrs_t_angle` | Spatial planar angle between QRS and T axes | Frontal plane | < 45° (normal), > 100° (pathology) |
| 24 | `sokolow_lyon` | Left Ventricular Hypertrophy index: $S_{V1} + R_{V5}$ | Leads V1 & V5 | < 3.5 mV (normal) |

---

## 2. Signal Preprocessing & Delineation Algorithm

### 2.1 Bandpass Filtering
Each raw lead signal $s_l(t)$ is cleaned using a 4th-order zero-phase Butterworth filter:
- Passband: $[0.5\,\text{Hz}, 40.0\,\text{Hz}]$
- Sampling Rate: $f_s = 500\,\text{Hz}$

### 2.2 Continuous Wavelet Transform (CWT) Delineation
To eliminate boundary inflation caused by Discrete Wavelet Transform (DWT), the system employs CWT (`method="cwt"` via NeuroKit2):
1. **R-Peak Detection**: Detected primarily on Lead II using gradient peak enhancement.
2. **CWT Wave Boundary Localization**: For each beat $k$ with R-peak timestamp $r_k$, scale-space decomposition identifies:
   - P-wave onset $P_{\text{on}}$, peak $P_{\text{peak}}$, and offset $P_{\text{off}}$ within $r_k + [-150, -10]\,\text{ms}$
   - QRS onset $Q_{\text{on}}$, Q-peak $Q_{\text{peak}}$, S-peak $S_{\text{peak}}$, and QRS offset $S_{\text{off}}$ within $r_k + [-50, +50]\,\text{ms}$
   - T-wave onset $T_{\text{on}}$, peak $T_{\text{peak}}$, and offset $T_{\text{off}}$ within $r_k + [+30, +250]\,\text{ms}$
3. **Isoelectric Baseline Estimation**: Calculated from the median PR segment ($[P_{\text{off}}, Q_{\text{on}}]$) across all valid beats in the 10-second recording.

---

## 3. Leakage-Free Preprocessing & Vector Construction

### 3.1 Patient-Wise Stratified Partitioning
- Dataset is partitioned into Train (70%), Validation (10%), and Test (20%) sets grouped by `patient_id`.
- **Guarantee**: Zero patient overlap ($0\%$) between splits.

### 3.2 Imputation and Standardization
Let $X \in \mathbb{R}^{N \times 24}$ denote the raw extracted clinical feature matrix:
1. **Missing Indicator Matrix**: $M_{i,j} = \mathbb{I}(X_{i,j} \text{ is NaN}) \in \{0, 1\}^{N \times 24}$
2. **Median Imputation**:
   $$\mu_j = \text{median}(\{X_{i,j} \mid i \in \mathcal{D}_{\text{train}}, X_{i,j} \neq \text{NaN}\})$$
   $$\tilde{X}_{i,j} = \begin{cases} X_{i,j} & \text{if } M_{i,j} = 0 \\ \mu_j & \text{if } M_{i,j} = 1 \end{cases}$$
3. **Z-Score Normalization**:
   $$\bar{x}_j = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{i \in \mathcal{D}_{\text{train}}} \tilde{X}_{i,j}, \quad \sigma_j = \sqrt{\frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{i \in \mathcal{D}_{\text{train}}} (\tilde{X}_{i,j} - \bar{x}_j)^2}$$
   $$Z_{i,j} = \frac{\tilde{X}_{i,j} - \bar{x}_j}{\sigma_j + \epsilon}$$
4. **Final Model Input Vector**:
   $$v_i = [Z_{i, 1..24} \parallel M_{i, 1..24}] \in \mathbb{R}^{48}$$

---

## 4. Model Architectures

```
+-----------------------------------------------------------------------------------+
| 1. Attention MLP Autoencoder (Params: 205,789)                                    |
| Input (48-D) -> BatchNorm1d -> Linear(48,256) -> ReLU -> Dropout(0.3)             |
|              -> Linear(256,128) -> ResidualBlock(128)                             |
|              -> MultiheadAttention(dim=128, heads=4)                              |
|              -> Linear(128,64) -> ReLU -> Linear(64,32)  --> Latent z (32-D)      |
| Decoder:        Latent (32-D) -> Linear(32,64) -> Linear(64,128)                  |
|                               -> Linear(128,256) -> Linear(256,24) -> Recon (24-D)|
| Classifier:     Latent (32-D) -> Linear(32,5) -> Diagnostic Logits (5-D)          |
+-----------------------------------------------------------------------------------+
| 2. Beta-VAE Autoencoder (Params: 108,829, Beta = 0.5 - 1.0)                       |
| Input (48-D) -> BatchNorm1d -> Linear(48,256) -> ReLU -> Linear(256,128)          |
|              -> Linear(128,64) -> Linear(64,32)                                   |
|              -> [fc_mu(32,32), fc_logvar(32,32)]                                  |
| Latent:         z = mu + eps * exp(0.5 * logvar), eps ~ N(0, I) (32-D)            |
| Decoder:        z -> Linear(32,64) -> Linear(64,128) -> Linear(128,256)           |
|                   -> Linear(256,24) -> Recon (24-D)                              |
| Classifier:     mu -> Linear(32,5) -> Diagnostic Logits (5-D)                     |
+-----------------------------------------------------------------------------------+
| 3. FT-Transformer Autoencoder (Params: 74,173)                                    |
| Input (48-D) -> FeatureTokenizer(48 features -> 48 x 64-D tokens)                 |
|              -> Prepend learnable [CLS] token (1 x 64-D) -> Sequence (49 x 64-D)  |
|              -> TransformerEncoder(layers=2, heads=4, d_model=64, ffn=128)        |
|              -> Extract [CLS] Token (64-D) -> Linear(64,64) -> ReLU               |
|              -> Linear(64,32) --> Latent z (32-D)                                 |
| Decoder:        Latent (32-D) -> Linear(32,64) -> Linear(64,128)                  |
|                               -> Linear(128,256) -> Linear(256,24) -> Recon (24-D)|
| Classifier:     Latent (32-D) -> Linear(32,5) -> Diagnostic Logits (5-D)          |
+-----------------------------------------------------------------------------------+
```

---

## 5. Loss Formulation & Training Procedure

### 5.1 Multi-Task Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}}(x, \hat{x}) + \beta \mathcal{L}_{\text{KLD}}(\mu, \sigma) + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}}(y, \hat{y})$$

Where:
- $\mathcal{L}_{\text{recon}} = \frac{1}{24} \sum_{j=1}^{24} (x_j - \hat{x}_j)^2$
- $\mathcal{L}_{\text{KLD}} = -\frac{1}{2 \times 32} \sum_{k=1}^{32} (1 + \log \sigma_k^2 - \mu_k^2 - \sigma_k^2)$ (applied only to Beta-VAE)
- $\mathcal{L}_{\text{cls}} = \frac{1}{5} \sum_{c=1}^{5} \text{BCEWithLogits}(y_c, \hat{y}_c)$ with positive class weights $w_c = \frac{N_{\text{neg}, c}}{N_{\text{pos}, c}}$
- Weighting parameter $\lambda_{\text{cls}} = 1.0$, $\beta = 0.5$.

### 5.2 Optimizer & Regularization
- **Optimizer**: AdamW ($\text{lr} = 1\times 10^{-3}$, $\text{weight\_decay} = 1\times 10^{-4}$)
- **Scheduler**: `ReduceLROnPlateau(mode='min', factor=0.5, patience=5)`
- **Early Stopping**: 15 epochs without validation loss improvement.
- **Batch Size**: 128 (Train), 256 (Val/Test).

---

## 6. Evaluation Harness & Downstream Validation

1. **Downstream Diagnostic Classifier**:
   - Class-weighted multi-label Logistic Regression fitted on latent embeddings $z \in \mathbb{R}^{32}$.
   - Optimal decision thresholds $\tau_c \in [0.1, 0.9]$ tuned on the validation set for each diagnostic class (NORM, MI, STTC, CD, HYP).
2. **Latent Representation Quality Metrics**:
   - **Reconstruction MSE & MAE**
   - **Latent Dimension Variance**: $\text{Var}(z_k)$ across the test set.
   - **Dimension Redundancy**: Pearson correlation matrix $R_{j,k} = \text{corr}(z_j, z_k)$, counting highly correlated pairs ($|R| > 0.8$).
   - **Latent Collapse**: Number of latent dimensions with $\text{std}(z_k) < 0.01$.
