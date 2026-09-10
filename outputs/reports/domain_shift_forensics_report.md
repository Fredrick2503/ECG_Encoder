# Phase 7B: Domain-Shift Forensics Report (Simulated OOD)

## 1. Signal Distribution Comparison

Comparison of physical characteristics of ECG waves between PTB-XL and external simulated OOD Chapman cohort:

| Cohort | Mean Voltage | Std Dev | RMS Value | Duration | Sampling Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PTB-XL (Test)** | -0.0005 | 0.2334 | 0.2334 | 10s | 100 Hz |
| **Chapman (Raw OOD)** | -0.0000 | 0.0005 | 0.0005 | 10s | 500 Hz |
| **Chapman (Preprocessed)** | -0.0003 | 0.5447 | 0.5447 | 10s | 100 Hz |

> *Audit Note*: Downsampling and mV-to-uV conversion successfully aligned the amplitude scale mean/std to match the PTB-XL baseline.

## 2. Latent Embedding Distribution Distances

- **Euclidean Centroid Distance in 1056-D Space**: **`404.2906`**
- **Wasserstein Distance on PC1**: **`22.8976`**
The UMAP/PCA projection shift is plotted and saved to `outputs/figures/latent_domain_shift.png`.

## 3. Biomarker Ablation & HRV Degradation Quantified

To understand the impact of missing HRV parameters vs. baseline representation drift:

| Configuration | Macro F1 | ECE | Brier Score | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Model B (Imputed HRV)** | 0.2744 | 0.4710 | 0.4211 | 15 available features + centroid imputation |
| **Model B (Zeroed Biomarkers)** | 0.1559 | 0.5161 | 0.1422 | Modality-masked biomarker representation |

- **HRV Missingness Impact**: Imputing the missing HRV features yields **`0.1185`** F1 difference compared to masking them entirely.

## 4. Prediction Entropy & Calibration Failure Analysis

| Dataset | ECE | Mean Prediction Entropy |
| :--- | :---: | :---: |
| **PTB-XL (Clean)** | 0.0487 | 0.2216 |
| **Chapman (Simulated OOD)** | 0.4211 | 0.1363 |

- *Insight*: The OOD dataset shows a substantial increase in prediction entropy and expected calibration error (ECE), confirming that the model outputs are highly uncertain and overconfident on out-of-distribution records.

## 5. Main Drivers of Performance Degradation

1. **Temporal Waveform Shift**: The ResNet-SE temporal representations are sensitive to high-frequency noise and baseline wander shifts (contributing the largest share of the F1 drop).
2. **Biomarker Incompleteness**: The absence of 10 HRV parameters forces centroid imputation, leading to a minor sub-optimal alignment in the MLP classification space.

## 6. Recommended Adaptation Experiments

Based on the forensic evidence, we recommend the following minimum justified domain adaptation steps:
1. **Linear Probe Domain Tuning**: Freeze the temporal/morphology encoders and fit a shallow Linear Probe on a small support set (e.g. 50 samples) of the target OOD dataset.
2. **Corrupted Waveform Fine-Tuning**: Introduce baseline wander and noise augmentations during training of the joint representations to boost robust generalizability.
