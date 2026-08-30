# Modality-Dropout & Redundancy Benchmark Report

This report evaluates the performance of the classification engine under modality-dropout (zeroing out representation slices of T, M, or B).

## 1. Modality Redundancy Matrix

| Modality Combination   |   Macro F1 |   Macro AUC |   Subset Acc |   NORM F1 |    MI F1 |   STTC F1 |    CD F1 |   HYP F1 |   Brier Score |
|:-----------------------|-----------:|------------:|-------------:|----------:|---------:|----------:|---------:|---------:|--------------:|
| T + M + B (Full)       |   0.722079 |    0.909252 |    0.563333  |  0.873239 | 0.6875   |  0.658228 | 0.746269 | 0.645161 |     0.0848458 |
| T + M                  |   0.708128 |    0.911206 |    0.55      |  0.847826 | 0.674699 |  0.654088 | 0.779412 | 0.584615 |     0.0841227 |
| T + B                  |   0.708602 |    0.902619 |    0.566667  |  0.867797 | 0.662722 |  0.734375 | 0.696296 | 0.581818 |     0.0894261 |
| M + B                  |   0.528572 |    0.823304 |    0.276667  |  0.697479 | 0.468531 |  0.518072 | 0.521277 | 0.4375   |     0.133344  |
| T only                 |   0.708562 |    0.908084 |    0.553333  |  0.867133 | 0.670455 |  0.738462 | 0.680556 | 0.586207 |     0.0865032 |
| M only                 |   0.412718 |    0.806026 |    0.113333  |  0.123288 | 0.404432 |  0.518519 | 0.531915 | 0.485437 |     0.15381   |
| B only                 |   0.236668 |    0.629463 |    0.0133333 |  0.240506 | 0.391185 |  0        | 0.351648 | 0.2      |     0.186238  |

## 2. Key Findings

1. **Modality Complementarity**: The fully integrated model (T+M+B) achieves the highest F1 score (`0.722`). Removing the biomarker modality (T+M) only drops the F1 score slightly to `0.708`, while removing the morphology modality (T+B) drops it to `0.708`. This indicates significant classification redundancy and overlap between the modalities.
2. **Temporal Dominance**: Removing the temporal modality (M+B) results in a severe drop to F1 `0.528`, showing that the temporal features provide the core classification information for the majority of labels.
