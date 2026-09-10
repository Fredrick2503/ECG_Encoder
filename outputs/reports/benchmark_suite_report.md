# Benchmark Experiments Suite Report

## Summary of Evaluated Architectures and Datasets

| Architecture / Method | Target Dataset | Test Accuracy | Test Macro F1 | Test ROC-AUC | Test Sensitivity | Time (s) | Checkpoint |
|---|---|---|---|---|---|---|---|
| **CNN-LSTM** | MIT-BIH | 83.56% | 42.29% | 50.00% | 47.96% | 15.7s | `benchmark_cnn_lstm_best.pt` |
| **ECGFormer** | MIT-BIH | 82.35% | 46.01% | 50.00% | 54.46% | 16.6s | `benchmark_ecgformer_best.pt` |
| **CNN-Transformer** | MIT-BIH | 79.45% | 44.71% | 50.00% | 52.74% | 19.7s | `benchmark_cnn_transformer_best.pt` |
| **Hybrid BERT-CNN** | MIT-BIH | 88.07% | 49.07% | 50.00% | 55.38% | 32.4s | `benchmark_hybrid_bert_cnn_best.pt` |
| **FoundationalECGNet** | PTB-XL + CinC | 59.06% | 65.06% | 90.55% | 59.76% | 4047.4s | `benchmark_foundationalecgnet_best.pt` |
| **RR Interval AF Detection** | MIT-BIH AF | 88.69% | 82.76% | 92.09% | 79.88% | 20.9s | `benchmark_rr_interval_af_detection_best.pt` |

## Analysis & Findings

1. **MIT-BIH Arrhythmia Benchmarks:** Evaluated AAMI 5-class beat classification across CNN-LSTM, ECGFormer, CNN-Transformer, and Hybrid BERT-CNN under standard inter-patient evaluation protocols.
2. **FoundationalECGNet:** Evaluated 12-lead multi-label cardiac diagnosis on PTB-XL using hierarchical SE-ResNet temporal backbone and bidirectional cross-lead attention.
3. **RR Interval AF Detection:** Assessed RR interval sequence dynamics and statistical HRV features on MIT-BIH AF rhythm classification.

All best model checkpoints are preserved in `models/`.