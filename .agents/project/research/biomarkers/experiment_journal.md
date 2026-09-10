# Biomarker Encoder Experiment Journal

This journal tracks specific experimental trials, ablation runs, and evaluation benchmarks for the Biomarker Encoder module.

---

## Trial Summary Table

| Trial ID | Date | Delineation | Preprocessing | Architecture | Recon MSE | Downstream Macro F1 | Downstream ROC-AUC | Status |
|---|---|---|---|---|---|---|---|---|
| **EXP-BIO-001** | 2026-08-18 | DWT | Global Split (Leaked) | Attention MLP | 0.1705 | 0.6486 | 0.8668 | Superseded (Leakage) |
| **EXP-BIO-002** | 2026-08-18 | DWT | Global Split (Leaked) | Beta-VAE | 0.3891 | 0.6445 | 0.8666 | Superseded (Leakage) |
| **EXP-BIO-003** | 2026-08-18 | DWT | Global Split (Leaked) | FT-Transformer | 0.1387 | 0.6343 | 0.8605 | Superseded (Leakage) |
| **EXP-BIO-004** | 2026-08-19 | DWT | Leak-Free Split | Attention MLP | 0.1752 | 0.6442 | 0.8655 | Validated (DWT Bias) |
| **EXP-BIO-005** | 2026-08-19 | DWT | Leak-Free Split | Beta-VAE | 0.3980 | 0.6445 | 0.8623 | Validated (DWT Bias) |
| **EXP-BIO-006** | 2026-08-19 | DWT | Leak-Free Split | FT-Transformer | 0.1412 | 0.6360 | 0.8589 | Validated (DWT Bias) |
| **EXP-BIO-007** | 2026-08-20 | CWT | Leak-Free Split | Attention MLP | 0.1813 | 0.6332 | **0.8622** | **Production Checkpoint** |
| **EXP-BIO-008** | 2026-08-20 | CWT | Leak-Free Split | Beta-VAE | 0.4130 | **0.6347** | 0.8607 | **Production Checkpoint** |
| **EXP-BIO-009** | 2026-08-20 | CWT | Leak-Free Split | FT-Transformer | **0.1286** | 0.6192 | 0.8575 | **Production Checkpoint** |

---

## Detailed Trial Notes

### EXP-BIO-007: Attention MLP with CWT & Leak-Free Preprocessing
- **Configuration**: 205k parameters, 4 attention heads, hidden dim = 128, latent dim = 32, dropout = 0.3, weight decay = $1\times 10^{-4}$.
- **Reconstruction Performance**: $\text{MSE} = 0.1813$, $\text{MAE} = 0.2950$.
- **Latent Diagnostics**: 0 collapsed dimensions, total variance = 99.55, mean absolute correlation = 0.2452.
- **Downstream Multi-Label Classification**:
  - Macro F1: **0.6332** | Weighted F1: **0.6737** | Macro ROC-AUC: **0.8622** | Macro PR-AUC: **0.6876**
  - Per-Class F1: NORM: 0.8023, MI: 0.6281, STTC: 0.6127, CD: 0.6140, HYP: 0.5087.

### EXP-BIO-008: Beta-VAE with CWT & Leak-Free Preprocessing
- **Configuration**: 108k parameters, hidden dim = 128, latent dim = 32, $\beta = 0.5$, weight decay = $1\times 10^{-4}$.
- **Reconstruction Performance**: $\text{MSE} = 0.4130$, $\text{MAE} = 0.4654$.
- **Latent Diagnostics**: 0 collapsed dimensions, total variance = 3.93, mean absolute correlation = 0.3146.
- **Downstream Multi-Label Classification**:
  - Macro F1: **0.6347** | Weighted F1: **0.6767** | Macro ROC-AUC: **0.8607** | Macro PR-AUC: **0.6918**
  - Per-Class F1: NORM: 0.8112, MI: 0.6327, STTC: 0.6213, CD: 0.6000, HYP: 0.5081.

### EXP-BIO-009: FT-Transformer with CWT & Leak-Free Preprocessing
- **Configuration**: 74k parameters, 2 Transformer layers, 4 heads, $d_{\text{model}} = 64$, FFN dim = 128, learnable [CLS] token.
- **Reconstruction Performance**: $\text{MSE} = 0.1286$, $\text{MAE} = 0.2423$ (Lowest Reconstruction Error).
- **Latent Diagnostics**: 0 collapsed dimensions, total variance = 22.96, zero pairs with $|r| > 0.8$.
- **Downstream Multi-Label Classification**:
  - Macro F1: **0.6192** | Weighted F1: **0.6608** | Macro ROC-AUC: **0.8575** | Macro PR-AUC: **0.6786**
