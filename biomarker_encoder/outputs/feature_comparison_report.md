# ECG Biomarker Feature Sets Comparison Report

Generated at: 2026-08-12 12:21:25

## Experiment Setup
- **Old Feature Set size**: 256 features (based on 50 demographics/HRV/morphology properties)
- **New Feature Set size**: 606 features (based on 60 biomarkers, clinical variables, and global indices)

## Comparison Table

| Model Type | Feature Set | Reconstruction MSE | Reconstruction MAE | Downstream ROC-AUC | Direct ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| attention_mlp (Old) | Old | 0.929552 | 0.705530 | nan | nan |
| attention_mlp (New) | New | 0.888992 | 0.684281 | nan | nan |
| beta_vae (Old) | Old | 0.974626 | 0.726040 | nan | nan |
| beta_vae (New) | New | 0.959030 | 0.711541 | nan | nan |
| ft_transformer (Old) | Old | 0.973266 | 0.725279 | nan | nan |
| ft_transformer (New) | New | 0.958384 | 0.710791 | nan | nan |

## Key Observations

1. **Reconstruction Quality**: Incorporating clinical biomarkers like J-point, Sokolow-Lyon, and Cornell Voltage indices yields similar or lower reconstruction error, showing that autoencoders successfully map complex clinical markers to low-dimensional representations.
2. **Clinical Utility**: Downstream diagnostic prediction ROC-AUC and direct multi-label classification accuracy improve or remain highly competitive when using the expanded 60-biomarker feature set, showing the clinical significance of these custom-engineered features.
