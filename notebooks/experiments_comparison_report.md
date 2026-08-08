# ECG Foundation Representation - Balancing Experiments Report

This report compares the performance of 4 distinct data balancing / filtering experiments on the PTB-XL dataset.
Evaluation completed on: `2026-08-06 20:50:07`. Total duration: `2792.7s`.

## 1. Overview Comparison Table

| Experiment / Mode | Val Size | Test Size | ResNet W | Trans. W | Subset Acc | Macro F1 | Macro AUC | Hamming Loss |
|-------------------|----------|-----------|----------|----------|------------|----------|-----------|--------------|
| `average` | 1714 | 1712 | 0.10 | 0.90 | 0.4141 | 0.6492 | 0.8500 | 0.1921 |
| `max` | 1801 | 1812 | 0.30 | 0.70 | 0.4349 | 0.6644 | 0.8620 | 0.1858 |
| `min` | 1529 | 1486 | 0.40 | 0.60 | 0.3634 | 0.6638 | 0.8570 | 0.2011 |
| `binary` | 2193 | 2203 | 0.00 | 1.00 | 0.7208 | 0.7942 | 0.8658 | 0.2170 |

---

## 2. Detailed Per-Class Breakdown per Experiment

### Experiment: `average`

| Class | Threshold | Precision | Recall | F1-Score | ROC-AUC | Support |
|-------|-----------|-----------|--------|----------|---------|---------|
| NORM | 0.4951 | 0.5971 | 0.8584 | 0.7042 | 0.9049 | 473 |
| MI | 0.2476 | 0.5621 | 0.7125 | 0.6284 | 0.7921 | 553 |
| STTC | 0.4357 | 0.7348 | 0.7146 | 0.7246 | 0.8895 | 508 |
| CD | 0.1189 | 0.6667 | 0.6667 | 0.6667 | 0.8400 | 468 |
| HYP | 0.1585 | 0.5231 | 0.5211 | 0.5221 | 0.8236 | 261 |

### Experiment: `max`

| Class | Threshold | Precision | Recall | F1-Score | ROC-AUC | Support |
|-------|-----------|-----------|--------|----------|---------|---------|
| NORM | 0.5445 | 0.6697 | 0.8883 | 0.7637 | 0.9211 | 573 |
| MI | 0.2575 | 0.5510 | 0.7125 | 0.6215 | 0.8057 | 553 |
| STTC | 0.3070 | 0.7071 | 0.7461 | 0.7261 | 0.8972 | 508 |
| CD | 0.2080 | 0.6320 | 0.6427 | 0.6373 | 0.8302 | 473 |
| HYP | 0.2080 | 0.5074 | 0.6590 | 0.5733 | 0.8559 | 261 |

### Experiment: `min`

| Class | Threshold | Precision | Recall | F1-Score | ROC-AUC | Support |
|-------|-----------|-----------|--------|----------|---------|---------|
| NORM | 0.3961 | 0.5335 | 0.8057 | 0.6419 | 0.9069 | 247 |
| MI | 0.3466 | 0.6140 | 0.7161 | 0.6611 | 0.8040 | 553 |
| STTC | 0.1585 | 0.6204 | 0.8622 | 0.7216 | 0.8713 | 508 |
| CD | 0.2575 | 0.6714 | 0.7137 | 0.6919 | 0.8606 | 461 |
| HYP | 0.1486 | 0.5394 | 0.6820 | 0.6024 | 0.8423 | 261 |

### Experiment: `binary`

| Class | Threshold | Precision | Recall | F1-Score | ROC-AUC | Support |
|-------|-----------|-----------|--------|----------|---------|---------|
| NORM | 0.5445 | 0.7043 | 0.8653 | 0.7766 | 0.8658 | 958 |
| ABNORM | 0.1486 | 0.7946 | 0.8297 | 0.8118 | 0.8657 | 1245 |

