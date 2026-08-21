# Age 18-30 Cohort Comprehensive Audit Report

## 1. Split & Patient Separation Audit
- Filtered Train Records: **140**
- Filtered Val Records: **25**
- Filtered Test Records: **22**
- Train vs Val Patient Overlaps: **0**
- Train vs Test Patient Overlaps: **0**
- Val vs Test Patient Overlaps: **0**
  *Verification Verdict*: **SUCCESS** (disjoint partitions verified).

## 2. Statistical Signficance Comparison (T+M vs T+M+B)
- Wilcoxon Signed-Rank p-value: **`0.052020`**
- Permutation Test p-value: **`0.060000`**

### Model Performance Panel

| Model / Metric | Macro F1 | Macro AUC | Subset Accuracy | Macro ECE | Brier Score |
| --- | --- | --- | --- | --- | --- |
| **Model A (T+M)** | 0.5953 | nan | 0.7727 | 0.3093 | 0.1322 |
| **Model B (T+M+B)** | 0.7953 | nan | 0.8182 | 0.3692 | 0.1695 |

## 3. Robustness & Lead Loss Analysis (18-30 Cohort)
- **Clean Baseline F1:** `0.7953`
- **High-Frequency Noise F1:** `0.7953`
- **Chest-Leads Masked F1:** `0.7953`
