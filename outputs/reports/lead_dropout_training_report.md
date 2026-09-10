# Lead-Dropout Robustness Training Report

This report presents results from training the MLP classifier with randomized lead masking during training to improve clinical robustness.

## 1. 5-Seed Evaluation of Lead-Dropout Training

|   Seed |   Clean F1 |   Clean AUC |   Mask V5 F1 |   Mask V5 HYP F1 |   Mask Chest F1 |
|-------:|-----------:|------------:|-------------:|-----------------:|----------------:|
|     42 |   0.708575 |    0.910009 |     0.646439 |         0.315789 |        0.612239 |
|     43 |   0.720705 |    0.910257 |     0.656501 |         0.27027  |        0.621675 |
|     44 |   0.723824 |    0.905642 |     0.657984 |         0.315789 |        0.608017 |
|     45 |   0.724895 |    0.907545 |     0.668782 |         0.358974 |        0.5992   |
|     46 |   0.71398  |    0.91324  |     0.646521 |         0.315789 |        0.578422 |

## 2. Average Robustness Metrics vs. Baseline Model B

| Model | Clean Macro F1 | Mask V5 Macro F1 | Mask V5 HYP F1 | Mask Chest F1 |
| --- | --- | --- | --- | --- |
| **Baseline Model B** | `0.722079` | `0.608964` | `0.171429` | `0.337853` |
| **Lead-Dropout Trained (Mean)** | `0.718396` | `0.655246` | `0.315323` | `0.603911` |

## 3. Analysis & Verdict

Training with randomized lead masking allows the model to learn alternative pathways and dependencies across the standard 12 leads, dramatically improving performance when key diagnostic leads are missing or disconnected. Specifically, Left Ventricular Hypertrophy (HYP) classification F1 under V5 masking improved from `0.171429` to `0.315323`, representing a massive robustness improvement while maintaining high clean-performance.
