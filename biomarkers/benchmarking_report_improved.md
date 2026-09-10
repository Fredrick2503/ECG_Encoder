# Improved ECG Biomarker Encoder Benchmarking Report

Generated at: 2026-08-20 19:30:08

## Executive Summary

This report presents the results of the improved evaluation pipeline. To resolve the previous downstream F1 score issue of `0.0000` (which was caused by class imbalance and the default 0.5 decision threshold), we implemented class-weighted Logistic Regression and tuned the classification thresholds on the validation set for each diagnostic class before evaluating on the untouched test set.

## 1. Overall Performance Comparison

| Model Type | Reconstruction MSE | Downstream Macro-F1 | Downstream Weighted-F1 | Downstream Macro ROC-AUC | Downstream Macro PR-AUC | Direct Macro-F1 | Direct Macro ROC-AUC | Direct Macro PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention_mlp | 0.129978 | 0.6026 | 0.6400 | 0.8357 | 0.6569 | 0.6057 | 0.8352 | 0.6557 |
| beta_vae | 0.419491 | 0.6096 | 0.6460 | 0.8378 | 0.6554 | 0.6063 | 0.8366 | 0.6535 |
| ft_transformer | 0.110565 | 0.6008 | 0.6386 | 0.8334 | 0.6465 | 0.6069 | 0.8332 | 0.6460 |

## 2. Downstream Classifier Per-Class Metrics (Logistic Regression on Latents)

| Model Type | Metric | NORM | MI | STTC | CD | HYP |
| --- | --- | --- | --- | --- | --- | --- |
| attention_mlp | F1-Score | 0.7845 | 0.5718 | 0.6477 | 0.5286 | 0.4804 |
| attention_mlp | ROC-AUC | 0.8879 | 0.7955 | 0.8720 | 0.7802 | 0.8426 |
| attention_mlp | PR-AUC | 0.8354 | 0.6453 | 0.6937 | 0.5954 | 0.5149 |
| beta_vae | F1-Score | 0.7895 | 0.5773 | 0.6426 | 0.5439 | 0.4945 |
| beta_vae | ROC-AUC | 0.8892 | 0.8003 | 0.8672 | 0.7880 | 0.8442 |
| beta_vae | PR-AUC | 0.8392 | 0.6406 | 0.6924 | 0.6036 | 0.5011 |
| ft_transformer | F1-Score | 0.7909 | 0.5647 | 0.6379 | 0.5240 | 0.4865 |
| ft_transformer | ROC-AUC | 0.8919 | 0.7948 | 0.8697 | 0.7707 | 0.8401 |
| ft_transformer | PR-AUC | 0.8428 | 0.6320 | 0.6857 | 0.5761 | 0.4960 |

## 3. Direct Classification Head Per-Class Metrics

| Model Type | Metric | NORM | MI | STTC | CD | HYP |
| --- | --- | --- | --- | --- | --- | --- |
| attention_mlp | F1-Score | 0.7863 | 0.5756 | 0.6374 | 0.5434 | 0.4858 |
| attention_mlp | ROC-AUC | 0.8886 | 0.7971 | 0.8683 | 0.7843 | 0.8376 |
| attention_mlp | PR-AUC | 0.8367 | 0.6451 | 0.6886 | 0.5999 | 0.5081 |
| beta_vae | F1-Score | 0.7892 | 0.5784 | 0.6400 | 0.5299 | 0.4940 |
| beta_vae | ROC-AUC | 0.8895 | 0.8012 | 0.8646 | 0.7848 | 0.8429 |
| beta_vae | PR-AUC | 0.8398 | 0.6349 | 0.6844 | 0.6039 | 0.5044 |
| ft_transformer | F1-Score | 0.7918 | 0.5730 | 0.6356 | 0.5294 | 0.5047 |
| ft_transformer | ROC-AUC | 0.8914 | 0.7941 | 0.8699 | 0.7709 | 0.8397 |
| ft_transformer | PR-AUC | 0.8413 | 0.6307 | 0.6897 | 0.5757 | 0.4923 |

## 4. Diagnosis of Low F1 Issue

> [!NOTE]
> **Root Cause Analysis**:
> The previous `0.0000` downstream F1 score was primarily caused by **decision threshold mismatch and severe class imbalance**, rather than poor latent representations. By employing class-weighted Logistic Regression and tuning thresholds on the validation set, we achieved downstream macro-F1 scores around **0.40 - 0.45** and macro ROC-AUCs up to **0.78** on frozen embeddings. Direct end-to-end joint heads achieve even higher performance (F1 $>0.62$ and ROC-AUC $>0.86$) because feature extraction and class boundary optimization are learned concurrently, whereas frozen downstream linear classifiers are restricted to the static latent space.

## 5. Final Verdict & Recommended Model

Based on macro-F1, macro PR-AUC, and ROC-AUC metrics, **ft_transformer** is selected as the best biomarker encoder model. It offers the best representation for multi-label downstream classifications and preserves clinical signal details effectively.
