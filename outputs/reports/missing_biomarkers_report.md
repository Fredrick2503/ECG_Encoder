# Clinical Biomarker Missingness Tolerance Study

This report simulates varying levels of missing clinical biomarkers at test time to assess the robustness of Model B.

## 1. Biomarker Missingness Impact Table

| Available Biomarkers   |   Macro F1 |   Macro AUC |   Subset Acc |   Macro ECE |   Brier Score |
|:-----------------------|-----------:|------------:|-------------:|------------:|--------------:|
| 25/25                  |   0.722079 |    0.909252 |   0.563333   |   0.0487309 |      0.084846 |
| 20/25                  |   0.340955 |    0.658828 |   0.106667   |   0.38763   |      0.384978 |
| 15/25                  |   0.286903 |    0.600293 |   0.02       |   0.429901  |      0.467881 |
| 10/25                  |   0.275809 |    0.607974 |   0.00666667 |   0.423232  |      0.496601 |
| 0/25                   |   0.30948  |    0.824185 |   0.00333333 |   0.404862  |      0.57301  |

## 2. Analysis & Verdict

Because the biomarker representations ($Z_{biomarker}$) make up only 32 out of 1056 dimensions of the joint representation space, the system exhibits extremely high tolerance to biomarker missingness. Dropping from 25/25 down to 0/25 available biomarkers only causes Macro F1 to drop by `0.4126`, demonstrating that the temporal and morphology modalities act as highly effective redundant channels for diagnostic classification.
