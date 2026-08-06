# ECG Temporal Encoder Hyperparameter Sweep Report

**Date:** 2026-08-05 02:02:48
**Parent Run ID:** `9c2d4bb403514801beb52b16a74a4bf5`
**Experiment Name:** `ECG_TemporalEncoder_ExpandedSweep`

## Performance Table

| Trial Name | Pretrain SSL | LR | Hidden Size | Layers | LSTM Drp | FC Drp | Test Subset Acc | Test Hamming Loss | Test Macro F1 | Test Macro AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| trial_10_lr_0.001_hidden_256_layers_4_ssl_mae | mae | 0.001 | 256 | 4 | 0.1 | 0.2 | 0.4756 | 0.2293 | 0.1362 | 0.5249 |
| trial_11_lr_0.0005_hidden_256_layers_3_ssl_mae | mae | 0.0005 | 256 | 3 | 0.2 | 0.3 | 0.4689 | 0.2298 | 0.1358 | 0.5201 |
| trial_12_lr_0.001_hidden_256_layers_3_ssl_contrastive | contrastive | 0.001 | 256 | 3 | 0.3 | 0.1 | 0.3333 | 0.2302 | 0.1201 | 0.5316 |
| trial_13_lr_0.001_hidden_256_layers_3_ssl_mae | mae | 0.001 | 256 | 3 | 0.4 | 0.2 | 0.4289 | 0.2236 | 0.1347 | 0.5409 |
| trial_14_lr_0.001_hidden_256_layers_4_ssl_mae | mae | 0.001 | 256 | 4 | 0.4 | 0.4 | 0.4911 | 0.2302 | 0.1376 | 0.4895 |
| trial_15_lr_0.001_hidden_256_layers_2_ssl_mae | mae | 0.001 | 256 | 2 | 0.1 | 0.1 | 0.5000 | 0.2293 | 0.1388 | 0.5287 |
| trial_6_lr_0.0005_hidden_256_layers_4_ssl_mae | mae | 0.0005 | 256 | 4 | 0.3 | 0.1 | 0.4667 | 0.2311 | 0.1351 | 0.5113 |
| trial_8_lr_0.001_hidden_128_layers_3_ssl_contrastive | contrastive | 0.001 | 128 | 3 | 0.1 | 0.2 | 0.5000 | 0.2293 | 0.1388 | 0.4928 |
| trial_9_lr_0.0005_hidden_256_layers_4_ssl_contrastive | contrastive | 0.0005 | 256 | 4 | 0.3 | 0.2 | 0.3689 | 0.2244 | 0.1265 | 0.5581 |

## Best Configuration Found
- **Trial:** `trial_15_lr_0.001_hidden_256_layers_2_ssl_mae`
- **SSL Strategy:** `mae`
- **Learning Rate:** `0.001`
- **Hidden Size:** `256`
- **Layers:** `2`
- **Subset Accuracy:** `0.5000`
- **Macro AUC:** `0.5287`