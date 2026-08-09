# Ensemble Model Per-Class Metrics

**Optimal Ensemble Configuration:** ResNet Weight = `0.70`, Transformer Weight = `0.30`

## 1. Ensemble with Class-Specific Optimized Thresholds on Test Set

| Class | Optimized Threshold | Precision | Recall | F1-Score | ROC-AUC | Support |
|-------|---------------------|-----------|--------|----------|---------|---------|
| NORM | 0.5049 | 0.7998 | 0.9367 | 0.8629 | 0.9486 | 964 |
| MI | 0.4456 | 0.7421 | 0.7233 | 0.7326 | 0.9201 | 553 |
| STTC | 0.4258 | 0.7357 | 0.8110 | 0.7715 | 0.9338 | 508 |
| CD | 0.4159 | 0.7450 | 0.7510 | 0.7480 | 0.9151 | 498 |
| HYP | 0.4060 | 0.5880 | 0.5970 | 0.5925 | 0.8985 | 263 |

## 2. Ensemble with Default 0.5 Threshold on Test Set

| Class | Threshold | Precision | Recall | F1-Score | ROC-AUC | Support |
|-------|-----------|-----------|--------|----------|---------|---------|
| NORM | 0.5000 | 0.7984 | 0.9367 | 0.8621 | 0.9486 | 964 |
| MI | 0.5000 | 0.7564 | 0.6908 | 0.7221 | 0.9201 | 553 |
| STTC | 0.5000 | 0.7635 | 0.7500 | 0.7567 | 0.9338 | 508 |
| CD | 0.5000 | 0.7812 | 0.7169 | 0.7476 | 0.9151 | 498 |
| HYP | 0.5000 | 0.6618 | 0.5209 | 0.5830 | 0.8985 | 263 |

Evaluation completed in 261.68 seconds.