# MasterMind State

## Loop Identity

**Loop ID:** mastermind_loop_001  
**Started:** 2026-08-09 13:12  
**Last Updated:** 2026-08-14 07:03 (IST)  
**Status:** COMPLETED — All Benchmark Suites B/C/D/E/E2 Finished

---

## Target Metrics

| Metric | Target | Current Best | Gap |
|---|---|---|---|
| ROC-AUC (macro) | ≥ 0.95 | 0.8818 (E1-8 Fusion) | 0.0682 |
| Macro F1 | ≥ 0.75 | 0.6851 (E1-8 Fusion) | 0.0649 |
| Subset Accuracy | ≥ 0.65 | 0.5933 (D2-5) | 0.0567 |

**Target Reached?** No — gap remains. Targets are ambitious for 2K subset; full PTB-XL (21K) scaling is next.

---

## Trial Budget

| Parameter | Value |
|---|---|
| Max Trials | 35 + B6 + C17 + D6 + E8 + E2 |
| Suites Completed | B, C, D (D0+D2), E (E1-0–E1-8), E2 |
| Remaining Budget | N/A (all planned suites exhausted) |
| Max Barrier Retries | 3 |
| Consecutive Barrier Count | 0 |

---

## Best Trial So Far (by Metric)

| Metric | Best Trial | Value |
|---|---|---|
| ROC-AUC | **E1-8** (C5+D2-5 Fusion α=0.80) | **0.8818** |
| Macro F1 | **E1-8** (C5+D2-5 Fusion α=0.80) | **0.6851** |
| Subset Accuracy | **D2-5** (CBLoss + MI/STTC+CD Aux) | **0.5933** |
| kNN Purity | **E2-2** (Joint Concatenated) | **0.5991** |
| Linear Probe AUC | **E2-2** (Joint Concatenated) | **0.7331** |

### Final Locked Model: **E1-8**
- Architecture: C5 (CBLoss ResNet-SE) + D2-5 (CBLoss+Aux ResNet-SE) fusion at α=0.80
- ROC-AUC: **0.8818** | Macro F1: **0.6851** | Subset Acc: **0.5833**
- Macro ECE (Platt calibration): **0.0532**

---

## Pause & Resume Control Panel

- **Last Completed Suite:** `E2` (Representation-Quality Validation)
- **Last Completed Trial:** `E2-3-Joint` (kNN on Joint Concatenated)
- **Next Steps:** Morphology Encoder implementation (MS-11). Scale to full PTB-XL (21K).

---

## Sync Log

- **EDA Report:** [eda_class_distribution.md](file:///c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/eda_class_distribution.md)
- **Suite B Report:** [benchmark_b_results.md](file:///c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/benchmark_b_results.md)
- **Suite C Report:** [benchmark_c_results.md](file:///c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/benchmark_c_results.md)
- **Suite D Report:** [benchmark_d_results.md](file:///c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/benchmark_d_results.md)
- **Suite D0 Diagnosis:** [suite_d_phase_d0_diagnosis.md](file:///c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/suite_d_phase_d0_diagnosis.md)
- **Suite E Report:** [benchmark_e_results.md](file:///c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/benchmark_e_results.md)
- **Suite E2 Report:** [representation_validation_results.md](file:///c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/representation_validation_results.md)
- **Thesis Notes:** [thesis_notes.md](file:///c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/docs/thesis_notes.md)
- **Experiment Journal:** [experiment_journal.md](file:///c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/.agents/project/research/mastermind/experiment_journal.md)

---

*Maintained by @mastermind using the `experiment-orchestrator` skill.*
