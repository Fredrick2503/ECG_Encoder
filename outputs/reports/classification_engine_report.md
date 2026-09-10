# Phase 4: Unified Classification Engine Benchmark Report

**Dataset Used**: PTB-XL 2K Subset

## 1. Classifier Model Comparison

| model           | loss   |   macro_auc |   macro_f1 |   subset_acc |   macro_ece |
|:----------------|:-------|------------:|-----------:|-------------:|------------:|
| C0_Linear_Probe | BCE    |    0.91024  |   0.709967 |     0.543333 |    0.056384 |
| C1_MLP          | BCE    |    0.912036 |   0.712924 |     0.546667 |    0.04256  |
| C1_MLP          | CB-BCE |    0.903978 |   0.708762 |     0.55     |    0.053323 |
| C1_MLP          | ASL    |    0.900409 |   0.688264 |     0.523333 |    0.230118 |

## 2. Representation Ablation Study (Linear Probe)

| ablation            |   macro_auc |   macro_f1 |   subset_acc |   macro_ece |
|:--------------------|------------:|-----------:|-------------:|------------:|
| Temporal_Only (T)   |    0.917009 |   0.691422 |     0.556667 |   0.0471048 |
| Morphology_Only (M) |    0.832409 |   0.602917 |     0.446667 |   0.0746944 |
| Biomarker_Only (B)  |    0.854814 |   0.606857 |     0.456667 |   0.0385165 |
| Pairwise_T_M        |    0.912454 |   0.719353 |     0.576667 |   0.0520423 |
| Pairwise_T_B        |    0.91696  |   0.709763 |     0.56     |   0.0522961 |
| Pairwise_M_B        |    0.862162 |   0.646332 |     0.516667 |   0.0585736 |
| Full_Fused (T_M_B)  |    0.909829 |   0.715312 |     0.536667 |   0.0578781 |

## 3. Best Model Per-Class Detailed Performance

**Model**: C1 MLP Classifier | **Loss**: BCE

| Class   |   F1-Score |   ROC-AUC |       ECE |   Threshold |
|:--------|-----------:|----------:|----------:|------------:|
| NORM    |   0.879195 |  0.934978 | 0.0557197 |        0.36 |
| MI      |   0.68323  |  0.904472 | 0.0378905 |        0.23 |
| STTC    |   0.670807 |  0.935387 | 0.0366782 |        0.21 |
| CD      |   0.734375 |  0.911017 | 0.0492023 |        0.46 |
| HYP     |   0.597015 |  0.874325 | 0.0333095 |        0.3  |