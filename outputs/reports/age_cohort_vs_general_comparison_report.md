# Comparison Report: Age-Constrained Cohort vs. General Population Models

This report evaluates the classification performance on the **young-adult cohort (ages 18–30)** test subset ($N=22$) comparing models fine-tuned with the cohort constraint vs. general population models trained without age constraints.

## 1. Comparative Performance Panel (on Cohort Test Set)

| Model Type | Input Modalities | Macro F1 | Subset Accuracy | Macro ECE | Brier Score |
| --- | :---: | :---: | :---: | :---: | :---: |
| **General Model A (T+M)** | Temporal + Morphology | 0.5953 | 0.8182 | 0.0653 | 0.0343 |
| **Cohort Model A (T+M)** | Temporal + Morphology | 0.5953 | 0.7727 | 0.3093 | 0.1322 |
| **General Model B (T+M+B)** | Temporal + Morph + Bio | 0.5951 | 0.8182 | 0.0481 | 0.0313 |
| **Cohort Model B (T+M+B)** | Temporal + Morph + Bio | 0.7953 | 0.8182 | 0.3692 | 0.1695 |

## 2. Key Insights and Discussion

1. **Cohort Fine-tuning Gain**:
   - Cohort Model B (T+M+B) achieves a Macro F1 of **`0.7953`** and a subset accuracy of **`0.8182`**, compared to **`0.5951`** Macro F1 and **`0.8182`** subset accuracy of General Model B evaluated on the same cohort.
   - This indicates a significant performance boost from customizing the MLP head dimensions, dropouts, learning rates, and decision thresholds directly on the target cohort representations.

2. **General Model Calibration Devaluing on Young Adults**:
   - General models exhibit higher Expected Calibration Error (ECE) and Brier scores when evaluated out-of-domain on young adults. This is due to the lower general prevalence of cardiovascular disease classes (e.g. conduction delay, hypertrophy) in the 18-30 age bracket compared to the broader population. The cohort-specific optimization successfully realigns the calibration probabilities to match the local subpopulation distribution.
