# EDA — Class Distribution Analysis

**Generated:** 2026-08-09 19:09  
**Dataset:** PTB-XL (100Hz low-res)  
**Training subset:** 1000 records  

## Class Counts per Split

| Class | Train N | Train % | Val N | Test N | Pos Weight (inv-freq) |
|---|---|---|---|---|---|
| **NORM** | 548 | 54.8% | 91 | 105 | 0.82 |
| **MI** | 173 | 17.3% | 45 | 38 | 4.78 |
| **STTC** | 212 | 21.2% | 56 | 42 | 3.72 |
| **CD** | 205 | 20.5% | 36 | 39 | 3.88 |
| **HYP** | 108 | 10.8% | 27 | 22 | 8.26 |

**Imbalance Ratio (max:min):** 5.1x  

## Interpretation

- Imbalance ratio > 5x → class-weighted loss recommended.
- Imbalance ratio > 10x → ASL or CBLoss strongly recommended (Ridnik 2021, Cui 2019).
- Classes with pos_weight > 5.0 are significantly underrepresented.
- Training without compensation leads to model collapse toward dominant classes.

## Selected Pos Weights for Weighted BCE Experiments

```
  NORM: 0.8248
  MI: 4.7803
  STTC: 3.7170
  CD: 3.8780
  HYP: 8.2593
```
