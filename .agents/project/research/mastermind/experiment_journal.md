# Experiment Journal — MasterMind Loop

This file records all experiment trials conducted under the MasterMind
autonomous loop. Each entry follows the experiment trial template.

**Template:** `.agents/project/research/_templates/experiment_trial_template.md`

---

## Summary Table

| Trial | Architecture | Strategy | Loss | ROC-AUC | Macro F1 | Status | Date |
|---|---|---|---|---|---|---|---|
| *(pre-loop)* | BiLSTM | Reconstruction | BCE | ~0.85 | — | Reference | — |
| *(pre-loop)* | BiLSTM | MAE | BCE | ~0.857 | — | Reference | — |
| *(pre-loop)* | ECGTransformer | MAE | BCE | ~0.872 | — | Reference | — |
| *(pre-loop)* | ECGResNet1D+SE | — | ASL | — | — | Reference | — |
| *(pre-loop)* | ResNet+Transformer | Ensemble | ASL | 0.8918 | — | Best Known | — |

---

## Pre-Loop Baseline Reference

The following results are known from prior experiments (before MasterMind loop):

- **Baseline BiLSTM + MAE:** ROC-AUC ≈ 0.857 (from run_comparison_experiment.py)
- **ECGTransformer + MAE:** ROC-AUC ≈ 0.872 (13.11% subset acc improvement)
- **Goal-Oriented Search best:** ROC-AUC = 0.8921 (peak val), 0.8918 (test)
- **Ensemble (ResNet+SE + Transformer + ASL):** ROC-AUC = 0.8918 (test)
- **Target:** ROC-AUC ≥ 0.92

**Gap to target:** ~0.0282

---

## Trial Entries

*(Entries will be appended here by @experiment-logger after each trial.)*

---

*Maintained by @experiment-logger using the `experiment-file-sync` skill.*

---

### Trial: T01_no_filter

**Date:** 2026-08-08 22:11  
**Status:** COMPLETED  
**MLflow Run ID:** `35c06777c78b45b291ff6ce9ba80290b`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | none |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 1 |
| Training Time | 7s |

**Reason for this configuration:**  
> Baseline: raw signal with only Z-score normalization. Establishes the absolute lower bound for preprocessing benefit.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.6356 |
| Macro F1 | 0.3795 |
| Subset Accuracy | 0.0000 |
| Hamming Loss | 0.7380 |
| Macro Sensitivity | 0.9875 |
| Macro Specificity | 0.0190 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.7359 | 0.6755 | 1.0000 | 0.0000 |
| MI | 0.3348 | 0.2804 | 0.9375 | 0.0952 |
| STTC | 0.6684 | 0.4127 | 1.0000 | 0.0000 |
| CD | 0.5690 | 0.3471 | 1.0000 | 0.0000 |
| HYP | 0.8700 | 0.1818 | 1.0000 | 0.0000 |

---

### Trial: T02_bandpass

**Date:** 2026-08-08 22:12  
**Status:** COMPLETED  
**MLflow Run ID:** `f5275fc4c68f4e22b082fe6df20498fe`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 1 |
| Training Time | 7s |

**Reason for this configuration:**  
> Standard clinical ECG preprocessing (0.5-40 Hz). Removes baseline wander (<0.5 Hz) and high-freq EMG noise (>40 Hz). Most commonly used in literature (Hannun et al., 2019; Ribeiro et al., 2020).

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.5835 |
| Macro F1 | 0.3898 |
| Subset Accuracy | 0.0100 |
| Hamming Loss | 0.7260 |
| Macro Sensitivity | 0.9922 |
| Macro Specificity | 0.0612 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.7807 | 0.7313 | 0.9608 | 0.3061 |
| MI | 0.4881 | 0.2759 | 1.0000 | 0.0000 |
| STTC | 0.6460 | 0.4127 | 1.0000 | 0.0000 |
| CD | 0.4051 | 0.3471 | 1.0000 | 0.0000 |
| HYP | 0.5978 | 0.1818 | 1.0000 | 0.0000 |

---

### Trial: T01_no_filter

**Date:** 2026-08-08 23:33  
**Status:** COMPLETED  
**MLflow Run ID:** `702075f51b7245918234e0c1e231de75`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | none |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 1 |
| Training Time | 8s |

**Reason for this configuration:**  
> Baseline: raw signal with only Z-score normalization. Establishes the absolute lower bound for preprocessing benefit.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.5788 |
| Macro F1 | 0.4115 |
| Subset Accuracy | 0.0000 |
| Hamming Loss | 0.5220 |
| Macro Sensitivity | 0.7931 |
| Macro Specificity | 0.3444 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.3701 | 0.6755 | 1.0000 | 0.0000 |
| MI | 0.5982 | 0.3226 | 0.6250 | 0.5714 |
| STTC | 0.6778 | 0.5195 | 0.7692 | 0.5811 |
| CD | 0.5992 | 0.3582 | 0.5714 | 0.5696 |
| HYP | 0.6489 | 0.1818 | 1.0000 | 0.0000 |

---

### Trial: T02_bandpass

**Date:** 2026-08-08 23:34  
**Status:** COMPLETED  
**MLflow Run ID:** `1737ae947f7a414c92b97f88d286c829`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 1 |
| Training Time | 7s |

**Reason for this configuration:**  
> Standard clinical ECG preprocessing (0.5-40 Hz). Removes baseline wander (<0.5 Hz) and high-freq EMG noise (>40 Hz). Most commonly used in literature (Hannun et al., 2019; Ribeiro et al., 2020).

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.7052 |
| Macro F1 | 0.4170 |
| Subset Accuracy | 0.0000 |
| Hamming Loss | 0.6460 |
| Macro Sensitivity | 0.9490 |
| Macro Specificity | 0.1617 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8319 | 0.6892 | 1.0000 | 0.0612 |
| MI | 0.5580 | 0.2752 | 0.9375 | 0.0714 |
| STTC | 0.7677 | 0.5915 | 0.8077 | 0.6757 |
| CD | 0.6426 | 0.3471 | 1.0000 | 0.0000 |
| HYP | 0.7256 | 0.1818 | 1.0000 | 0.0000 |

---

### Trial: T01_no_filter

**Date:** 2026-08-08 23:56  
**Status:** COMPLETED  
**MLflow Run ID:** `ef74bea48ce9462da1f58f34093830ab`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | none |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 14 |
| Training Time | 758s |

**Reason for this configuration:**  
> Baseline: raw signal with only Z-score normalization. Establishes the absolute lower bound for preprocessing benefit.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8593 |
| Macro F1 | 0.6247 |
| Subset Accuracy | 0.5233 |
| Hamming Loss | 0.1700 |
| Macro Sensitivity | 0.7096 |
| Macro Specificity | 0.8326 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8952 | 0.8353 | 0.9161 | 0.7034 |
| MI | 0.8822 | 0.6316 | 0.6207 | 0.9174 |
| STTC | 0.8976 | 0.6813 | 0.8857 | 0.7826 |
| CD | 0.8456 | 0.6042 | 0.5800 | 0.9320 |
| HYP | 0.7757 | 0.3711 | 0.5455 | 0.8277 |

---

### Trial: T02_bandpass

**Date:** 2026-08-09 00:12  
**Status:** COMPLETED  
**MLflow Run ID:** `dff0b8191bae4e9da782fdcec247c5d7`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 17 |
| Training Time | 899s |

**Reason for this configuration:**  
> Standard clinical ECG preprocessing (0.5-40 Hz). Removes baseline wander (<0.5 Hz) and high-freq EMG noise (>40 Hz). Most commonly used in literature (Hannun et al., 2019; Ribeiro et al., 2020).

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8721 |
| Macro F1 | 0.6385 |
| Subset Accuracy | 0.5800 |
| Hamming Loss | 0.1500 |
| Macro Sensitivity | 0.6812 |
| Macro Specificity | 0.8651 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.9119 | 0.8622 | 0.9484 | 0.7310 |
| MI | 0.8709 | 0.5660 | 0.5172 | 0.9256 |
| STTC | 0.9163 | 0.6951 | 0.8143 | 0.8391 |
| CD | 0.8679 | 0.6341 | 0.5200 | 0.9760 |
| HYP | 0.7932 | 0.4348 | 0.6061 | 0.8539 |

---

### Trial: T01_no_filter

**Date:** 2026-08-09 00:18  
**Status:** COMPLETED  
**MLflow Run ID:** `f2732ad877d84000971d95702eb2a439`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | none |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 1 |
| Training Time | 8s |

**Reason for this configuration:**  
> Baseline: raw signal with only Z-score normalization. Establishes the absolute lower bound for preprocessing benefit.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.5642 |
| Macro F1 | 0.3572 |
| Subset Accuracy | 0.0000 |
| Hamming Loss | 0.6320 |
| Macro Sensitivity | 0.8286 |
| Macro Specificity | 0.1975 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.6511 | 0.6755 | 1.0000 | 0.0000 |
| MI | 0.4100 | 0.2759 | 1.0000 | 0.0000 |
| STTC | 0.4917 | 0.4127 | 1.0000 | 0.0000 |
| CD | 0.5570 | 0.2400 | 0.1429 | 0.9873 |
| HYP | 0.7111 | 0.1818 | 1.0000 | 0.0000 |

---

### Trial: T02_bandpass

**Date:** 2026-08-09 00:18  
**Status:** COMPLETED  
**MLflow Run ID:** `e49a855564f74ef69fd27b787855e3a6`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 1 |
| Training Time | 8s |

**Reason for this configuration:**  
> Standard clinical ECG preprocessing (0.5-40 Hz). Removes baseline wander (<0.5 Hz) and high-freq EMG noise (>40 Hz). Most commonly used in literature (Hannun et al., 2019; Ribeiro et al., 2020).

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.3982 |
| Macro F1 | 0.3786 |
| Subset Accuracy | 0.0000 |
| Hamming Loss | 0.7520 |
| Macro Sensitivity | 1.0000 |
| Macro Specificity | 0.0000 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.5758 | 0.6755 | 1.0000 | 0.0000 |
| MI | 0.2909 | 0.2759 | 1.0000 | 0.0000 |
| STTC | 0.5296 | 0.4127 | 1.0000 | 0.0000 |
| CD | 0.4177 | 0.3471 | 1.0000 | 0.0000 |
| HYP | 0.1767 | 0.1818 | 1.0000 | 0.0000 |

---

### Trial: G001_transformer_none

**Date:** 2026-08-09 00:24  
**Status:** COMPLETED  
**MLflow Run ID:** `b55f393f1c0c45aab8c6c9db8508645b`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | none |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 1 |
| Training Time | 8s |

**Reason for this configuration:**  
> Baseline: raw signal with only Z-score normalization. Establishes the absolute lower bound for preprocessing benefit.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.6084 |
| Macro F1 | 0.3822 |
| Subset Accuracy | 0.0000 |
| Hamming Loss | 0.7400 |
| Macro Sensitivity | 1.0000 |
| Macro Specificity | 0.0152 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.7155 | 0.6755 | 1.0000 | 0.0000 |
| MI | 0.5707 | 0.2759 | 1.0000 | 0.0000 |
| STTC | 0.6346 | 0.4127 | 1.0000 | 0.0000 |
| CD | 0.7221 | 0.3652 | 1.0000 | 0.0759 |
| HYP | 0.3989 | 0.1818 | 1.0000 | 0.0000 |

---

### Trial: G002_resnet_se_none

**Date:** 2026-08-09 00:25  
**Status:** COMPLETED  
**MLflow Run ID:** `ecf2a9ca3aea4f078000aea39d44a58e`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | resnet_se |
| Filter Config | none |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 1 |
| Training Time | 10s |

**Reason for this configuration:**  
> Baseline: raw signal with only Z-score normalization. Establishes the absolute lower bound for preprocessing benefit.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.7349 |
| Macro F1 | 0.3425 |
| Subset Accuracy | 0.0000 |
| Hamming Loss | 0.6340 |
| Macro Sensitivity | 0.8190 |
| Macro Specificity | 0.1975 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.7513 | 0.6755 | 1.0000 | 0.0000 |
| MI | 0.7359 | 0.2759 | 1.0000 | 0.0000 |
| STTC | 0.7703 | 0.4127 | 1.0000 | 0.0000 |
| CD | 0.7071 | 0.1667 | 0.0952 | 0.9873 |
| HYP | 0.7100 | 0.1818 | 1.0000 | 0.0000 |

---

### Trial: T01_no_filter

**Date:** 2026-08-09 00:30  
**Status:** COMPLETED  
**MLflow Run ID:** `6a1259ac20974dc5b678d655479583fe`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | none |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 1 |
| Training Time | 8s |

**Reason for this configuration:**  
> Baseline: raw signal with only Z-score normalization. Establishes the absolute lower bound for preprocessing benefit.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.6413 |
| Macro F1 | 0.4274 |
| Subset Accuracy | 0.0000 |
| Hamming Loss | 0.5960 |
| Macro Sensitivity | 0.9000 |
| Macro Specificity | 0.1882 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.7547 | 0.6846 | 1.0000 | 0.0408 |
| MI | 0.5469 | 0.2759 | 1.0000 | 0.0000 |
| STTC | 0.5785 | 0.4127 | 1.0000 | 0.0000 |
| CD | 0.4997 | 0.3471 | 1.0000 | 0.0000 |
| HYP | 0.8267 | 0.4167 | 0.5000 | 0.9000 |

---

### Trial: T02_bandpass

**Date:** 2026-08-09 00:31  
**Status:** COMPLETED  
**MLflow Run ID:** `8164765b27b748e5beb7f42b98e9e0b3`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 1 |
| Training Time | 8s |

**Reason for this configuration:**  
> Standard clinical ECG preprocessing (0.5-40 Hz). Removes baseline wander (<0.5 Hz) and high-freq EMG noise (>40 Hz). Most commonly used in literature (Hannun et al., 2019; Ribeiro et al., 2020).

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.6390 |
| Macro F1 | 0.3801 |
| Subset Accuracy | 0.0000 |
| Hamming Loss | 0.7460 |
| Macro Sensitivity | 1.0000 |
| Macro Specificity | 0.0073 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.6495 | 0.6755 | 1.0000 | 0.0000 |
| MI | 0.6362 | 0.2807 | 1.0000 | 0.0238 |
| STTC | 0.7126 | 0.4127 | 1.0000 | 0.0000 |
| CD | 0.4834 | 0.3500 | 1.0000 | 0.0127 |
| HYP | 0.7133 | 0.1818 | 1.0000 | 0.0000 |

---

### Trial: T01_no_filter

**Date:** 2026-08-09 13:04  
**Status:** COMPLETED  
**MLflow Run ID:** `4f7db522a5664982954f670628617160`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | none |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 1 |
| Training Time | 14s |

**Reason for this configuration:**  
> Baseline: raw signal with only Z-score normalization. Establishes the absolute lower bound for preprocessing benefit.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.5713 |
| Macro F1 | 0.3650 |
| Subset Accuracy | 0.0000 |
| Hamming Loss | 0.7573 |
| Macro Sensitivity | 0.9878 |
| Macro Specificity | 0.0086 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.7395 | 0.6486 | 1.0000 | 0.0000 |
| MI | 0.6228 | 0.4339 | 0.9762 | 0.0185 |
| STTC | 0.5080 | 0.3006 | 0.9630 | 0.0244 |
| CD | 0.5733 | 0.3051 | 1.0000 | 0.0000 |
| HYP | 0.4127 | 0.1366 | 1.0000 | 0.0000 |

---

### Trial: T02_bandpass

**Date:** 2026-08-09 13:05  
**Status:** COMPLETED  
**MLflow Run ID:** `e9f693eddaa8417bbe827eeea27ec811`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 1 |
| Training Time | 10s |

**Reason for this configuration:**  
> Standard clinical ECG preprocessing (0.5-40 Hz). Removes baseline wander (<0.5 Hz) and high-freq EMG noise (>40 Hz). Most commonly used in literature (Hannun et al., 2019; Ribeiro et al., 2020).

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.6453 |
| Macro F1 | 0.3678 |
| Subset Accuracy | 0.0000 |
| Hamming Loss | 0.7587 |
| Macro Sensitivity | 1.0000 |
| Macro Specificity | 0.0051 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8380 | 0.6545 | 1.0000 | 0.0256 |
| MI | 0.6347 | 0.4375 | 1.0000 | 0.0000 |
| STTC | 0.6902 | 0.3051 | 1.0000 | 0.0000 |
| CD | 0.4490 | 0.3051 | 1.0000 | 0.0000 |
| HYP | 0.6148 | 0.1366 | 1.0000 | 0.0000 |

---

### Trial: T01_no_filter

**Date:** 2026-08-09 13:10  
**Status:** COMPLETED  
**MLflow Run ID:** `9d8d1b2a3d85421c8eabb3d9a9080373`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | none |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 7 |
| Training Time | 164s |

**Reason for this configuration:**  
> Baseline: raw signal with only Z-score normalization. Establishes the absolute lower bound for preprocessing benefit.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8014 |
| Macro F1 | 0.5160 |
| Subset Accuracy | 0.4067 |
| Hamming Loss | 0.2253 |
| Macro Sensitivity | 0.6086 |
| Macro Specificity | 0.8046 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8743 | 0.7639 | 0.7639 | 0.7821 |
| MI | 0.7804 | 0.5591 | 0.6190 | 0.7685 |
| STTC | 0.7995 | 0.4912 | 0.5185 | 0.8699 |
| CD | 0.8335 | 0.5753 | 0.7778 | 0.7967 |
| HYP | 0.7194 | 0.1905 | 0.3636 | 0.8058 |

---

### Trial: T01_no_filter

**Date:** 2026-08-09 13:15  
**Status:** COMPLETED  
**MLflow Run ID:** `65bcae6fa54b40dca3553eea6ccfe399`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | none |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 8 |
| Training Time | 145s |

**Reason for this configuration:**  
> Baseline: raw signal with only Z-score normalization. Establishes the absolute lower bound for preprocessing benefit.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.7980 |
| Macro F1 | 0.5359 |
| Subset Accuracy | 0.4133 |
| Hamming Loss | 0.2400 |
| Macro Sensitivity | 0.6827 |
| Macro Specificity | 0.7597 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8673 | 0.7949 | 0.8611 | 0.7179 |
| MI | 0.7890 | 0.5524 | 0.6905 | 0.6852 |
| STTC | 0.8064 | 0.5161 | 0.5926 | 0.8455 |
| CD | 0.8374 | 0.6377 | 0.8148 | 0.8374 |
| HYP | 0.6900 | 0.1786 | 0.4545 | 0.7122 |

---

### Trial: T02_bandpass

**Date:** 2026-08-09 13:18  
**Status:** COMPLETED  
**MLflow Run ID:** `7d69bdbc8ebf42c481146b6012149bee`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 9 |
| Training Time | 173s |

**Reason for this configuration:**  
> Standard clinical ECG preprocessing (0.5-40 Hz). Removes baseline wander (<0.5 Hz) and high-freq EMG noise (>40 Hz). Most commonly used in literature (Hannun et al., 2019; Ribeiro et al., 2020).

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8090 |
| Macro F1 | 0.5053 |
| Subset Accuracy | 0.4467 |
| Hamming Loss | 0.2013 |
| Macro Sensitivity | 0.5608 |
| Macro Specificity | 0.8266 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.9005 | 0.8077 | 0.8750 | 0.7308 |
| MI | 0.7652 | 0.5361 | 0.6190 | 0.7315 |
| STTC | 0.8416 | 0.3810 | 0.2963 | 0.9431 |
| CD | 0.8293 | 0.5797 | 0.7407 | 0.8211 |
| HYP | 0.7083 | 0.2222 | 0.2727 | 0.9065 |

---

### Trial: T03_bandpass_notch

**Date:** 2026-08-09 13:22  
**Status:** COMPLETED  
**MLflow Run ID:** `536fe447e5b54587a66570111bd56637`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 8 |
| Training Time | 179s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8049 |
| Macro F1 | 0.5307 |
| Subset Accuracy | 0.4333 |
| Hamming Loss | 0.2120 |
| Macro Sensitivity | 0.6200 |
| Macro Specificity | 0.8024 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8880 | 0.8025 | 0.8750 | 0.7179 |
| MI | 0.7745 | 0.5532 | 0.6190 | 0.7593 |
| STTC | 0.8422 | 0.5517 | 0.5926 | 0.8780 |
| CD | 0.8470 | 0.5882 | 0.7407 | 0.8293 |
| HYP | 0.6730 | 0.1579 | 0.2727 | 0.8273 |

---

### Trial: T04_fir

**Date:** 2026-08-09 13:26  
**Status:** COMPLETED  
**MLflow Run ID:** `6578bed18174460c9d4cc0ec4ec47f94`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | fir |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 10 |
| Training Time | 235s |

**Reason for this configuration:**  
> FIR bandpass filter (zero-phase, linear phase response). Advantages: no phase distortion vs Butterworth IIR. Useful when phase fidelity matters for morphological analysis.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8139 |
| Macro F1 | 0.5081 |
| Subset Accuracy | 0.4400 |
| Hamming Loss | 0.2280 |
| Macro Sensitivity | 0.6051 |
| Macro Specificity | 0.7895 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8942 | 0.8000 | 0.8611 | 0.7308 |
| MI | 0.8036 | 0.5435 | 0.5952 | 0.7685 |
| STTC | 0.8323 | 0.5172 | 0.5556 | 0.8699 |
| CD | 0.8335 | 0.5333 | 0.7407 | 0.7724 |
| HYP | 0.7057 | 0.1463 | 0.2727 | 0.8058 |

---

### Trial: T05_wavelet

**Date:** 2026-08-09 13:30  
**Status:** COMPLETED  
**MLflow Run ID:** `a12e6875c31649eba3fb68a2a3912744`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | wavelet |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 10 |
| Training Time | 177s |

**Reason for this configuration:**  
> Wavelet denoising (Daubechies db4, 4 levels). Donoho-Johnstone universal threshold. Preserves QRS complex morphology better than frequency-domain filtering for beat-level features.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8261 |
| Macro F1 | 0.5280 |
| Subset Accuracy | 0.4200 |
| Hamming Loss | 0.2120 |
| Macro Sensitivity | 0.6021 |
| Macro Specificity | 0.8169 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8748 | 0.7838 | 0.8056 | 0.7692 |
| MI | 0.7959 | 0.5417 | 0.6190 | 0.7407 |
| STTC | 0.8052 | 0.4912 | 0.5185 | 0.8699 |
| CD | 0.8765 | 0.6129 | 0.7037 | 0.8699 |
| HYP | 0.7783 | 0.2105 | 0.3636 | 0.8345 |

---

### Trial: T06_full_stack

**Date:** 2026-08-09 13:34  
**Status:** COMPLETED  
**MLflow Run ID:** `eb5e4039ce264f4f8d4a72183c3b1918`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | full_stack |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 10 |
| Training Time | 210s |

**Reason for this configuration:**  
> Maximum denoising: Butterworth + Notch + Wavelet. Stacks all proven methods. Risk: may over-smooth subtle pathological waveforms. Trade-off experiment.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8119 |
| Macro F1 | 0.5027 |
| Subset Accuracy | 0.4467 |
| Hamming Loss | 0.2107 |
| Macro Sensitivity | 0.5695 |
| Macro Specificity | 0.8121 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.9033 | 0.7927 | 0.9028 | 0.6538 |
| MI | 0.8009 | 0.5366 | 0.5238 | 0.8333 |
| STTC | 0.8082 | 0.4000 | 0.3704 | 0.8943 |
| CD | 0.8440 | 0.6176 | 0.7778 | 0.8374 |
| HYP | 0.7031 | 0.1667 | 0.2727 | 0.8417 |

---

### Trial: T07_robust_norm

**Date:** 2026-08-09 13:37  
**Status:** COMPLETED  
**MLflow Run ID:** `d370dbd6e9524bafad10d7a1901643c1`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | robust_norm |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 7 |
| Training Time | 190s |

**Reason for this configuration:**  
> Bandpass + Notch + Robust scaler (median/IQR). Robust normalization is better than Z-score when signals have outlier artifact spikes, which is common in 12-lead ECG.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8023 |
| Macro F1 | 0.4699 |
| Subset Accuracy | 0.4333 |
| Hamming Loss | 0.2160 |
| Macro Sensitivity | 0.5223 |
| Macro Specificity | 0.8309 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8917 | 0.8082 | 0.8194 | 0.8077 |
| MI | 0.7630 | 0.4658 | 0.4048 | 0.8704 |
| STTC | 0.8269 | 0.5312 | 0.6296 | 0.8374 |
| CD | 0.7955 | 0.4675 | 0.6667 | 0.7398 |
| HYP | 0.7345 | 0.0769 | 0.0909 | 0.8993 |

---

### Trial: T08_balance_avg

**Date:** 2026-08-09 13:41  
**Status:** COMPLETED  
**MLflow Run ID:** `dcdfc285939f40679c3a2fd4c44340f4`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass_notch |
| Balance Mode | average |
| Loss Function | asl |
| Best Epoch | 10 |
| Training Time | 175s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8025 |
| Macro F1 | 0.6205 |
| Subset Accuracy | 0.2600 |
| Hamming Loss | 0.2480 |
| Macro Sensitivity | 0.7730 |
| Macro Specificity | 0.7299 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8984 | 0.7000 | 0.8974 | 0.7658 |
| MI | 0.7037 | 0.5957 | 0.7778 | 0.5312 |
| STTC | 0.8503 | 0.6476 | 0.8293 | 0.7248 |
| CD | 0.8655 | 0.7400 | 0.8605 | 0.8131 |
| HYP | 0.6945 | 0.4194 | 0.5000 | 0.8145 |

---

### Trial: T09_balance_max

**Date:** 2026-08-09 13:45  
**Status:** COMPLETED  
**MLflow Run ID:** `ac3c655e4abd4209a43bdad11d2fec14`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass_notch |
| Balance Mode | max |
| Loss Function | asl |
| Best Epoch | 10 |
| Training Time | 183s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8253 |
| Macro F1 | 0.6093 |
| Subset Accuracy | 0.3600 |
| Hamming Loss | 0.2133 |
| Macro Sensitivity | 0.6890 |
| Macro Specificity | 0.8155 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.9271 | 0.8077 | 0.7778 | 0.9167 |
| MI | 0.7870 | 0.5437 | 0.7000 | 0.6818 |
| STTC | 0.8460 | 0.6265 | 0.7027 | 0.8230 |
| CD | 0.8415 | 0.6753 | 0.7429 | 0.8609 |
| HYP | 0.7248 | 0.3934 | 0.5217 | 0.7953 |

---

### Trial: T10_balance_min

**Date:** 2026-08-09 13:50  
**Status:** COMPLETED  
**MLflow Run ID:** `dd7cce5fa32445fe92022328e95b0d11`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass_notch |
| Balance Mode | min |
| Loss Function | asl |
| Best Epoch | 7 |
| Training Time | 194s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8022 |
| Macro F1 | 0.5536 |
| Subset Accuracy | 0.1133 |
| Hamming Loss | 0.3333 |
| Macro Sensitivity | 0.7367 |
| Macro Specificity | 0.6418 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8266 | 0.5000 | 0.5000 | 0.8952 |
| MI | 0.8266 | 0.6809 | 0.8276 | 0.6196 |
| STTC | 0.8552 | 0.6727 | 0.7255 | 0.7778 |
| CD | 0.8049 | 0.5741 | 0.7045 | 0.6887 |
| HYP | 0.6977 | 0.3401 | 0.9259 | 0.2276 |

---

### Trial: T11_bce_weighted

**Date:** 2026-08-09 13:53  
**Status:** COMPLETED  
**MLflow Run ID:** `978e74b355374255bd226377fea5dbdd`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | bce |
| Best Epoch | 8 |
| Training Time | 187s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.7911 |
| Macro F1 | 0.4953 |
| Subset Accuracy | 0.4267 |
| Hamming Loss | 0.2240 |
| Macro Sensitivity | 0.5680 |
| Macro Specificity | 0.8042 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8823 | 0.8158 | 0.8611 | 0.7692 |
| MI | 0.7723 | 0.5208 | 0.5952 | 0.7315 |
| STTC | 0.7955 | 0.4444 | 0.4444 | 0.8780 |
| CD | 0.8404 | 0.5455 | 0.6667 | 0.8293 |
| HYP | 0.6651 | 0.1500 | 0.2727 | 0.8129 |

---

### Trial: T12_focal_g1

**Date:** 2026-08-09 13:57  
**Status:** COMPLETED  
**MLflow Run ID:** `b7edb20b04214fc998f1a5de070589ef`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | focal_g1 |
| Best Epoch | 10 |
| Training Time | 192s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.7736 |
| Macro F1 | 0.5079 |
| Subset Accuracy | 0.4333 |
| Hamming Loss | 0.2387 |
| Macro Sensitivity | 0.6375 |
| Macro Specificity | 0.7732 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8736 | 0.8101 | 0.8889 | 0.7179 |
| MI | 0.7447 | 0.5169 | 0.5476 | 0.7778 |
| STTC | 0.7645 | 0.5106 | 0.8889 | 0.6504 |
| CD | 0.8332 | 0.4889 | 0.4074 | 0.9431 |
| HYP | 0.6521 | 0.2128 | 0.4545 | 0.7770 |

---

### Trial: T13_focal_g2

**Date:** 2026-08-09 14:01  
**Status:** COMPLETED  
**MLflow Run ID:** `36515308866246b8bc0607316423e3cf`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | focal_g2 |
| Best Epoch | 10 |
| Training Time | 189s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8126 |
| Macro F1 | 0.5147 |
| Subset Accuracy | 0.4067 |
| Hamming Loss | 0.2067 |
| Macro Sensitivity | 0.5731 |
| Macro Specificity | 0.8338 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.9008 | 0.7973 | 0.8194 | 0.7821 |
| MI | 0.8186 | 0.5714 | 0.5714 | 0.8333 |
| STTC | 0.8169 | 0.4082 | 0.3704 | 0.9024 |
| CD | 0.8395 | 0.6061 | 0.7407 | 0.8455 |
| HYP | 0.6874 | 0.1905 | 0.3636 | 0.8058 |

---

### Trial: T14_focal_g3

**Date:** 2026-08-09 14:05  
**Status:** COMPLETED  
**MLflow Run ID:** `f566149e50d04319a8363f2c7399ca28`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | focal_g3 |
| Best Epoch | 10 |
| Training Time | 195s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8069 |
| Macro F1 | 0.4982 |
| Subset Accuracy | 0.3667 |
| Hamming Loss | 0.2720 |
| Macro Sensitivity | 0.6693 |
| Macro Specificity | 0.7362 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8935 | 0.8153 | 0.8889 | 0.7308 |
| MI | 0.7970 | 0.5417 | 0.6190 | 0.7407 |
| STTC | 0.7949 | 0.3750 | 0.4444 | 0.7967 |
| CD | 0.8434 | 0.5538 | 0.6667 | 0.8374 |
| HYP | 0.7057 | 0.2051 | 0.7273 | 0.5755 |

---

### Trial: T15_asl_hard

**Date:** 2026-08-09 14:08  
**Status:** COMPLETED  
**MLflow Run ID:** `4005a11500814c57bbb1c842921d552e`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | asl_hard |
| Best Epoch | 7 |
| Training Time | 191s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8035 |
| Macro F1 | 0.5058 |
| Subset Accuracy | 0.4200 |
| Hamming Loss | 0.2280 |
| Macro Sensitivity | 0.6087 |
| Macro Specificity | 0.7831 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8910 | 0.7950 | 0.8889 | 0.6795 |
| MI | 0.7811 | 0.5455 | 0.6429 | 0.7222 |
| STTC | 0.8025 | 0.4348 | 0.3704 | 0.9268 |
| CD | 0.8555 | 0.5676 | 0.7778 | 0.7886 |
| HYP | 0.6874 | 0.1860 | 0.3636 | 0.7986 |

---

### Trial: T16_resnet_se

**Date:** 2026-08-09 14:12  
**Status:** COMPLETED  
**MLflow Run ID:** `3b4208f3a013494298099e7b313267c3`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | resnet_se |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 8 |
| Training Time | 153s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8648 |
| Macro F1 | 0.5999 |
| Subset Accuracy | 0.5133 |
| Hamming Loss | 0.1680 |
| Macro Sensitivity | 0.7044 |
| Macro Specificity | 0.8429 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.9222 | 0.8414 | 0.8472 | 0.8462 |
| MI | 0.8677 | 0.6939 | 0.8095 | 0.7963 |
| STTC | 0.8573 | 0.5846 | 0.7037 | 0.8455 |
| CD | 0.8922 | 0.6486 | 0.8889 | 0.8130 |
| HYP | 0.7848 | 0.2308 | 0.2727 | 0.9137 |

---

### Trial: T17_multiscale_cnn

**Date:** 2026-08-09 14:14  
**Status:** COMPLETED  
**MLflow Run ID:** `cb7f1dff6de84f8c92ce5f42bf8ca02d`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | multiscale_cnn |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 10 |
| Training Time | 135s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.7128 |
| Macro F1 | 0.4601 |
| Subset Accuracy | 0.3533 |
| Hamming Loss | 0.3520 |
| Macro Sensitivity | 0.7166 |
| Macro Specificity | 0.6353 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8486 | 0.7867 | 0.8194 | 0.7564 |
| MI | 0.7397 | 0.5275 | 0.5714 | 0.7685 |
| STTC | 0.6742 | 0.3922 | 0.7407 | 0.5528 |
| CD | 0.7067 | 0.4190 | 0.8148 | 0.5447 |
| HYP | 0.5945 | 0.1750 | 0.6364 | 0.5540 |

---

### Trial: T17_bilstm_supervised

**Date:** 2026-08-09 14:20  
**Status:** COMPLETED  
**MLflow Run ID:** `d1aad636dcb747cbb0542a57b298b792`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | bilstm |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 10 |
| Training Time | 278s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.4433 |
| Macro F1 | 0.3678 |
| Subset Accuracy | 0.0000 |
| Hamming Loss | 0.7307 |
| Macro Sensitivity | 0.9763 |
| Macro Specificity | 0.0408 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.5228 | 0.6452 | 0.9722 | 0.0385 |
| MI | 0.4285 | 0.4375 | 1.0000 | 0.0000 |
| STTC | 0.3719 | 0.3051 | 1.0000 | 0.0000 |
| CD | 0.4151 | 0.3051 | 1.0000 | 0.0000 |
| HYP | 0.4781 | 0.1460 | 0.9091 | 0.1655 |

---

### Trial: T17_bilstm_ssl_mae

**Date:** 2026-08-09 14:37  
**Status:** COMPLETED  
**MLflow Run ID:** `6eab1897a2c64a809807e4379962fa17`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | bilstm |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 1 |
| Training Time | 803s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.4634 |
| Macro F1 | 0.3666 |
| Subset Accuracy | 0.0000 |
| Hamming Loss | 0.7613 |
| Macro Sensitivity | 1.0000 |
| Macro Specificity | 0.0000 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.4970 | 0.6486 | 1.0000 | 0.0000 |
| MI | 0.4336 | 0.4375 | 1.0000 | 0.0000 |
| STTC | 0.5378 | 0.3051 | 1.0000 | 0.0000 |
| CD | 0.4143 | 0.3051 | 1.0000 | 0.0000 |
| HYP | 0.4343 | 0.1366 | 1.0000 | 0.0000 |

---

### Trial: T17_bilstm_ssl_reconstruction

**Date:** 2026-08-09 14:45  
**Status:** COMPLETED  
**MLflow Run ID:** `cf9d0390afcf416b87d571dcd242efc4`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | bilstm |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 6 |
| Training Time | 266s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.4988 |
| Macro F1 | 0.3273 |
| Subset Accuracy | 0.0000 |
| Hamming Loss | 0.5707 |
| Macro Sensitivity | 0.7286 |
| Macro Specificity | 0.2569 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.5641 | 0.6486 | 1.0000 | 0.0000 |
| MI | 0.4793 | 0.3724 | 0.6429 | 0.2963 |
| STTC | 0.4929 | 0.3103 | 1.0000 | 0.0244 |
| CD | 0.5215 | 0.3051 | 1.0000 | 0.0000 |
| HYP | 0.4362 | 0.0000 | 0.0000 | 0.9640 |

---

### Trial: T17_bilstm_ssl_contrastive

**Date:** 2026-08-09 14:56  
**Status:** COMPLETED  
**MLflow Run ID:** `eb9cddc9e5694ac8a6a980bd5df8a3cb`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | bilstm |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 9 |
| Training Time | 300s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.5141 |
| Macro F1 | 0.3151 |
| Subset Accuracy | 0.0067 |
| Hamming Loss | 0.4907 |
| Macro Sensitivity | 0.5435 |
| Macro Specificity | 0.4924 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.6077 | 0.5772 | 0.5972 | 0.5641 |
| MI | 0.5470 | 0.4034 | 0.5714 | 0.5093 |
| STTC | 0.5248 | 0.1818 | 0.1852 | 0.8130 |
| CD | 0.4601 | 0.3051 | 1.0000 | 0.0000 |
| HYP | 0.4310 | 0.1081 | 0.3636 | 0.5755 |

---

### Trial: T17_bigru

**Date:** 2026-08-09 15:01  
**Status:** COMPLETED  
**MLflow Run ID:** `137372ad66474d878d6b699ea8200fb9`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | bigru |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 7 |
| Training Time | 304s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.4901 |
| Macro F1 | 0.3698 |
| Subset Accuracy | 0.0000 |
| Hamming Loss | 0.7240 |
| Macro Sensitivity | 0.9818 |
| Macro Specificity | 0.0417 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.5123 | 0.6486 | 1.0000 | 0.0000 |
| MI | 0.4605 | 0.4375 | 1.0000 | 0.0000 |
| STTC | 0.4158 | 0.3051 | 1.0000 | 0.0000 |
| CD | 0.5110 | 0.3051 | 1.0000 | 0.0000 |
| HYP | 0.5507 | 0.1527 | 0.9091 | 0.2086 |

---

### Trial: T17_attn_bilstm

**Date:** 2026-08-09 15:07  
**Status:** COMPLETED  
**MLflow Run ID:** `fb5ff110c70640c6a5728fbede2de438`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | attn_bilstm |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 7 |
| Training Time | 296s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.7596 |
| Macro F1 | 0.4487 |
| Subset Accuracy | 0.0000 |
| Hamming Loss | 0.5213 |
| Macro Sensitivity | 0.9230 |
| Macro Specificity | 0.3760 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8818 | 0.8129 | 0.8750 | 0.7436 |
| MI | 0.7326 | 0.5135 | 0.9048 | 0.3704 |
| STTC | 0.7416 | 0.3051 | 1.0000 | 0.0000 |
| CD | 0.8323 | 0.4545 | 0.9259 | 0.5285 |
| HYP | 0.6095 | 0.1575 | 0.9091 | 0.2374 |

---

### Trial: T17_cnn_lstm_trans

**Date:** 2026-08-09 15:13  
**Status:** COMPLETED  
**MLflow Run ID:** `b23ee85ce1fb4a45a69f4224cea96273`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | cnn_lstm_transformer |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 7 |
| Training Time | 271s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.7750 |
| Macro F1 | 0.4686 |
| Subset Accuracy | 0.3933 |
| Hamming Loss | 0.2880 |
| Macro Sensitivity | 0.6463 |
| Macro Specificity | 0.7087 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8873 | 0.7925 | 0.8750 | 0.6923 |
| MI | 0.7606 | 0.5510 | 0.6429 | 0.7315 |
| STTC | 0.7404 | 0.3548 | 0.4074 | 0.8049 |
| CD | 0.8248 | 0.4694 | 0.8519 | 0.6098 |
| HYP | 0.6619 | 0.1754 | 0.4545 | 0.7050 |

---

### Trial: T18_best_resnet_full_stack

**Date:** 2026-08-09 15:17  
**Status:** COMPLETED  
**MLflow Run ID:** `14138f427a754bf59d6633a94156d6b3`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | resnet_se |
| Filter Config | full_stack |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 4 |
| Training Time | 248s |

**Reason for this configuration:**  
> Maximum denoising: Butterworth + Notch + Wavelet. Stacks all proven methods. Risk: may over-smooth subtle pathological waveforms. Trade-off experiment.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8421 |
| Macro F1 | 0.5681 |
| Subset Accuracy | 0.4600 |
| Hamming Loss | 0.1813 |
| Macro Sensitivity | 0.6415 |
| Macro Specificity | 0.8445 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.9090 | 0.8366 | 0.8889 | 0.7821 |
| MI | 0.8302 | 0.5897 | 0.5476 | 0.8796 |
| STTC | 0.8829 | 0.5965 | 0.6296 | 0.8943 |
| CD | 0.8181 | 0.5676 | 0.7778 | 0.7886 |
| HYP | 0.7704 | 0.2500 | 0.3636 | 0.8777 |

---

### Trial: T19_transformer_wavelet_asl

**Date:** 2026-08-09 15:21  
**Status:** COMPLETED  
**MLflow Run ID:** `0c3e8408f35f441cb52b70fb48529bd5`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | wavelet |
| Balance Mode | none |
| Loss Function | asl_hard |
| Best Epoch | 10 |
| Training Time | 197s |

**Reason for this configuration:**  
> Wavelet denoising (Daubechies db4, 4 levels). Donoho-Johnstone universal threshold. Preserves QRS complex morphology better than frequency-domain filtering for beat-level features.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.7916 |
| Macro F1 | 0.5147 |
| Subset Accuracy | 0.3667 |
| Hamming Loss | 0.2693 |
| Macro Sensitivity | 0.6974 |
| Macro Specificity | 0.7180 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8932 | 0.8199 | 0.9167 | 0.7051 |
| MI | 0.7899 | 0.5500 | 0.7857 | 0.5833 |
| STTC | 0.7498 | 0.4062 | 0.4815 | 0.8049 |
| CD | 0.8314 | 0.6000 | 0.6667 | 0.8780 |
| HYP | 0.6939 | 0.1972 | 0.6364 | 0.6187 |

---

### Trial: T20_resnet_robust_norm

**Date:** 2026-08-09 17:25  
**Status:** COMPLETED  
**MLflow Run ID:** `5ee07b2c1a584c26aa2884eded01ab25`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | resnet_se |
| Filter Config | robust_norm |
| Balance Mode | average |
| Loss Function | asl |
| Best Epoch | 15 |
| Training Time | 1046s |

**Reason for this configuration:**  
> Bandpass + Notch + Robust scaler (median/IQR). Robust normalization is better than Z-score when signals have outlier artifact spikes, which is common in 12-lead ECG.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8367 |
| Macro F1 | 0.6331 |
| Subset Accuracy | 0.2767 |
| Hamming Loss | 0.2393 |
| Macro Sensitivity | 0.7793 |
| Macro Specificity | 0.7450 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8808 | 0.7179 | 0.8537 | 0.8028 |
| MI | 0.7827 | 0.6442 | 0.8431 | 0.6010 |
| STTC | 0.8650 | 0.6766 | 0.8000 | 0.7767 |
| CD | 0.8709 | 0.7014 | 0.7872 | 0.7913 |
| HYP | 0.7841 | 0.4255 | 0.6122 | 0.7530 |

---

### Trial: T20_resnet_robust_norm

**Date:** 2026-08-09 17:43  
**Status:** COMPLETED  
**MLflow Run ID:** `e89f1d5dede04c7bb7d14cf454ad7121`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | resnet_se |
| Filter Config | robust_norm |
| Balance Mode | average |
| Loss Function | asl |
| Best Epoch | 13 |
| Training Time | 275s |

**Reason for this configuration:**  
> Bandpass + Notch + Robust scaler (median/IQR). Robust normalization is better than Z-score when signals have outlier artifact spikes, which is common in 12-lead ECG.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8142 |
| Macro F1 | 0.5922 |
| Subset Accuracy | 0.2333 |
| Hamming Loss | 0.2467 |
| Macro Sensitivity | 0.7025 |
| Macro Specificity | 0.7423 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8960 | 0.7209 | 0.7949 | 0.8559 |
| MI | 0.7579 | 0.6173 | 0.9259 | 0.3958 |
| STTC | 0.8185 | 0.5882 | 0.7317 | 0.7156 |
| CD | 0.9000 | 0.7234 | 0.7907 | 0.8411 |
| HYP | 0.6985 | 0.3111 | 0.2692 | 0.9032 |

---

### Trial: T21_bce_inv_freq

**Date:** 2026-08-09 17:59  
**Status:** COMPLETED  
**MLflow Run ID:** `2618c36131a7447a8659ec24981cfa0b`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | bce_inv_freq |
| Best Epoch | 18 |
| Training Time | 882s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8156 |
| Macro F1 | 0.5138 |
| Subset Accuracy | 0.4400 |
| Hamming Loss | 0.1947 |
| Macro Sensitivity | 0.5753 |
| Macro Specificity | 0.8378 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8928 | 0.8163 | 0.8333 | 0.8077 |
| MI | 0.7912 | 0.6047 | 0.6190 | 0.8333 |
| STTC | 0.8570 | 0.5283 | 0.5185 | 0.9024 |
| CD | 0.8627 | 0.5366 | 0.8148 | 0.7317 |
| HYP | 0.6743 | 0.0833 | 0.0909 | 0.9137 |

---

### Trial: T22_bce_sqrt_freq

**Date:** 2026-08-09 18:10  
**Status:** COMPLETED  
**MLflow Run ID:** `a77f81a268f64eda9c179448fed5339a`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | bce_sqrt_freq |
| Best Epoch | 25 |
| Training Time | 612s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8287 |
| Macro F1 | 0.5481 |
| Subset Accuracy | 0.4267 |
| Hamming Loss | 0.2053 |
| Macro Sensitivity | 0.6565 |
| Macro Specificity | 0.8000 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8949 | 0.7927 | 0.9028 | 0.6538 |
| MI | 0.8148 | 0.5714 | 0.5714 | 0.8333 |
| STTC | 0.8546 | 0.5231 | 0.6296 | 0.8293 |
| CD | 0.8829 | 0.6111 | 0.8148 | 0.8130 |
| HYP | 0.6965 | 0.2424 | 0.3636 | 0.8705 |

---

### Trial: T23_bce_label_smooth

**Date:** 2026-08-09 18:18  
**Status:** COMPLETED  
**MLflow Run ID:** `bd5d3fa0ebf44245bda58eb6a0a59059`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | bce_label_smooth |
| Best Epoch | 22 |
| Training Time | 419s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8057 |
| Macro F1 | 0.5029 |
| Subset Accuracy | 0.4333 |
| Hamming Loss | 0.2120 |
| Macro Sensitivity | 0.5763 |
| Macro Specificity | 0.7999 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.9012 | 0.7811 | 0.9167 | 0.6026 |
| MI | 0.7776 | 0.5176 | 0.5238 | 0.8056 |
| STTC | 0.8449 | 0.4643 | 0.4815 | 0.8699 |
| CD | 0.8338 | 0.6087 | 0.7778 | 0.8293 |
| HYP | 0.6710 | 0.1429 | 0.1818 | 0.8921 |

---

### Trial: T24_cb_loss

**Date:** 2026-08-09 18:28  
**Status:** COMPLETED  
**MLflow Run ID:** `3da6843bb3a34e079fc75bd24a2365c4`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | cb_loss |
| Best Epoch | 12 |
| Training Time | 366s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.7888 |
| Macro F1 | 0.4920 |
| Subset Accuracy | 0.3600 |
| Hamming Loss | 0.3040 |
| Macro Sensitivity | 0.7332 |
| Macro Specificity | 0.6750 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8725 | 0.7821 | 0.8472 | 0.7051 |
| MI | 0.7784 | 0.5962 | 0.7381 | 0.7130 |
| STTC | 0.7456 | 0.3830 | 0.6667 | 0.6016 |
| CD | 0.8413 | 0.4615 | 0.7778 | 0.6504 |
| HYP | 0.7063 | 0.2373 | 0.6364 | 0.7050 |

---

### Trial: T25_ldam

**Date:** 2026-08-09 18:34  
**Status:** COMPLETED  
**MLflow Run ID:** `55a57e7f1e2941b1934ef569b4b8ead8`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | ldam |
| Best Epoch | 13 |
| Training Time | 331s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8036 |
| Macro F1 | 0.4744 |
| Subset Accuracy | 0.4267 |
| Hamming Loss | 0.2333 |
| Macro Sensitivity | 0.5506 |
| Macro Specificity | 0.7955 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8876 | 0.8052 | 0.8611 | 0.7436 |
| MI | 0.7729 | 0.5591 | 0.6190 | 0.7685 |
| STTC | 0.7958 | 0.3273 | 0.3333 | 0.8455 |
| CD | 0.8516 | 0.5373 | 0.6667 | 0.8211 |
| HYP | 0.7103 | 0.1429 | 0.2727 | 0.7986 |

---

### Trial: T26_resnet_cb_loss

**Date:** 2026-08-09 18:44  
**Status:** COMPLETED  
**MLflow Run ID:** `175f8e63f589400dba3dce018a8268c1`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | resnet_se |
| Filter Config | full_stack |
| Balance Mode | none |
| Loss Function | cb_loss |
| Best Epoch | 9 |
| Training Time | 493s |

**Reason for this configuration:**  
> Maximum denoising: Butterworth + Notch + Wavelet. Stacks all proven methods. Risk: may over-smooth subtle pathological waveforms. Trade-off experiment.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8550 |
| Macro F1 | 0.5953 |
| Subset Accuracy | 0.4733 |
| Hamming Loss | 0.1667 |
| Macro Sensitivity | 0.6522 |
| Macro Specificity | 0.8686 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.9047 | 0.8435 | 0.8611 | 0.8333 |
| MI | 0.8318 | 0.5952 | 0.5952 | 0.8426 |
| STTC | 0.8750 | 0.5600 | 0.5185 | 0.9268 |
| CD | 0.8931 | 0.6349 | 0.7407 | 0.8699 |
| HYP | 0.7704 | 0.3429 | 0.5455 | 0.8705 |

---

### Trial: T27_resnet_bal_cb

**Date:** 2026-08-09 18:52  
**Status:** COMPLETED  
**MLflow Run ID:** `44b6a3ba50dc454dbc7d7ea769c27373`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | resnet_se |
| Filter Config | bandpass_notch |
| Balance Mode | average |
| Loss Function | cb_loss |
| Best Epoch | 5 |
| Training Time | 373s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.7946 |
| Macro F1 | 0.5781 |
| Subset Accuracy | 0.3000 |
| Hamming Loss | 0.2480 |
| Macro Sensitivity | 0.6844 |
| Macro Specificity | 0.7573 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.9027 | 0.7059 | 0.9231 | 0.7568 |
| MI | 0.7346 | 0.5985 | 0.7593 | 0.5625 |
| STTC | 0.8429 | 0.6173 | 0.6098 | 0.8624 |
| CD | 0.8965 | 0.7048 | 0.8605 | 0.7664 |
| HYP | 0.5965 | 0.2642 | 0.2692 | 0.8387 |

---

### Trial: T28_high_dropout

**Date:** 2026-08-09 18:59  
**Status:** COMPLETED  
**MLflow Run ID:** `307a73fbacfb4cfa861250ae6b055bce`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 15 |
| Training Time | 340s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.7996 |
| Macro F1 | 0.5093 |
| Subset Accuracy | 0.4000 |
| Hamming Loss | 0.2467 |
| Macro Sensitivity | 0.6487 |
| Macro Specificity | 0.7689 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8999 | 0.8112 | 0.8056 | 0.8333 |
| MI | 0.7817 | 0.5545 | 0.6667 | 0.7130 |
| STTC | 0.8103 | 0.4722 | 0.6296 | 0.7724 |
| CD | 0.8263 | 0.5385 | 0.7778 | 0.7561 |
| HYP | 0.6795 | 0.1702 | 0.3636 | 0.7698 |

---

### Trial: T29_heavy_wd

**Date:** 2026-08-09 19:04  
**Status:** COMPLETED  
**MLflow Run ID:** `eac81cae9fb04fcbae2b667c6c381b62`

**Configuration:**
| Parameter | Value |
|---|---|
| Architecture | transformer |
| Filter Config | bandpass_notch |
| Balance Mode | none |
| Loss Function | asl |
| Best Epoch | 15 |
| Training Time | 288s |

**Reason for this configuration:**  
> Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.

**Results:**
| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8084 |
| Macro F1 | 0.5327 |
| Subset Accuracy | 0.4200 |
| Hamming Loss | 0.2267 |
| Macro Sensitivity | 0.6518 |
| Macro Specificity | 0.7833 |

**Per-Class Metrics:**
| Class | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| NORM | 0.8948 | 0.8101 | 0.8889 | 0.7179 |
| MI | 0.7848 | 0.5474 | 0.6190 | 0.7500 |
| STTC | 0.8341 | 0.5385 | 0.5185 | 0.9106 |
| CD | 0.8672 | 0.5753 | 0.7778 | 0.7967 |
| HYP | 0.6612 | 0.1923 | 0.4545 | 0.7410 |
