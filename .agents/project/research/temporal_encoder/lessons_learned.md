# Temporal Encoder Lessons Learned

This document logs critical research insights, environment constraints, and pipeline optimizations discovered during the development of the Temporal Encoder module.

---

## 1. Self-Supervised pretraining and SimCLR Contrastive Loss
- **Insight:** Standard implementation of NT-Xent / SimCLR loss can suffer from slow loops and memory inefficiencies if computed iteratively.
- **Solution:** Formulate contrastive similarity calculation via a vectorized matrix multiplication of shape `(2B, 2B)`, setting self-similarity diagonals to zero with masks. This results in stable, fast, and highly parallel pretraining steps.
- **Insight:** In MAE pretraining, calculating MSE loss over the entire signal includes visible parts, which dilutes the reconstruction objective.
- **Solution:** Mask out the visible elements in the loss calculation and compute MSE strictly on indices where mask is zero (the masked regions).

---

## 2. MLflow Model Serialization Formats
- **Insight:** MLflow's default model logging can attempt to trace PyTorch models with the `'pt2'` format, which requires a dummy input example.
- **Solution:** Specify `serialization_format="pickle"` to use standard object serialization when an input example is not readily available or when the model format is recurrent (LSTMs).

---

## 3. Early Stopping and Dropout Regularization for LSTMs
- **Insight:** Training high-capacity BiLSTMs (e.g. hidden size 256) on sequential ECG signals can quickly result in overfitting, where validation loss starts to increase while training loss continues to drop.
- **Solution:** Implementing standard early stopping (patience 7) based on validation loss, paired with robust dropout (LSTM: 0.4, FC: 0.5) and a Plateau learning rate scheduler (decay factor 0.1, patience 3) effectively prevents overfitting and stabilizes convergence.
