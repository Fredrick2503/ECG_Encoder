# Research Lessons Learned

This document logs critical research insights, environment constraints, and pipeline optimizations discovered during the development of the ECG Foundation Representation System.

---

## 1. Environment Optimization under Network/Data Constraints
- **Insight:** Downloading massive deep learning libraries (like PyTorch ~2GB) or full datasets (PTB-XL ~1.7GB) can fail or time out on limited connections (e.g., mobile data), leading to corrupt files (e.g., `BadZipFile` exceptions) and pipeline halts.
- **Solution:**
  - **Lightweight Verification Mode:** Implementing a lightweight download option that fetches only metadata CSVs and the first record of each resolution (high-res 500Hz, low-res 100Hz, total ~10MB) directly from the PhysioNet HTTP files index. This allowed verification of loaders, splitting logic, and raw record reading in seconds.
  - **Conditional Dependency Guarding:** Packaging modules (like `data_management/__init__.py`) with `try-except` blocks for PyTorch-dependent classes. This allows data engineers and researchers to run loading and preprocessing tasks without needing a full PyTorch environment.
  - **HTTP Range Resuming:** Modifying the downloader to use `requests` with `Range` headers to support resuming interrupted downloads instead of starting over.

---

## 2. Zero-Phase Filtering for Cardiac Signals
- **Insight:** Direct lowpass/highpass/bandpass filters introduce phase distortion, which shifts peak locations (e.g. shifts R-peaks slightly to the right). This interferes with high-precision morphological segmentations like the Pan-Tompkins QRS algorithm.
- **Solution:** Always apply filtering using zero-phase forward-backward filtering (`scipy.signal.filtfilt`). This applies the filter coefficients in both directions, completely canceling phase shift.

---

## 3. Wavelet Denoising over Simple Smoothing
- **Insight:** Lowpass filtering removes high-frequency muscle noise but can also flatten the high-frequency peaks of QRS complexes, changing R-peak amplitude and QRS shape.
- **Solution:** Discrete Wavelet Transform (DWT) decomposition (e.g., using `db4` wavelet at level 4) with soft thresholding allows selective noise removal while preserving the sharp peaks of R-waves.

---

## 4. Multi-Label Resampling via Power-Set Mapping
- **Insight:** Standard class balancing algorithms (like SMOTE, ADASYN, and ENN) in `imbalanced-learn` do not support multi-hot label targets.
- **Solution:** Convert the multi-hot label matrix into unique integer class IDs using a label power-set mapping before resampling. Apply SMOTE-ENN on the resampled 2D flattened signals, and map the integer labels back to the original multi-hot label arrays.

---

## 5. Self-Supervised pretraining and SimCLR Contrastive Loss
- **Insight:** Standard implementation of NT-Xent / SimCLR loss can suffer from slow loops and memory inefficiencies if computed iteratively.
- **Solution:** Formulate contrastive similarity calculation via a vectorized matrix multiplication of shape `(2B, 2B)`, setting self-similarity diagonals to zero with masks. This results in stable, fast, and highly parallel pretraining steps.
- **Insight:** In MAE pretraining, calculating MSE loss over the entire signal includes visible parts, which dilutes the reconstruction objective.
- **Solution:** Mask out the visible elements in the loss calculation and compute MSE strictly on indices where mask is zero (the masked regions).

---

## 6. Directory Renaming vs. Slow Copying on Single Drive
- **Insight:** When using Python's `shutil.move` on existing directories, it falls back to copying files recursively one-by-one. For large datasets like PTB-XL (containing over 43,000 files), this fallback takes several minutes.
- **Solution:** Clean target directories first, then perform an atomic `os.rename` operation of the source folders, reducing the operation time from minutes to milliseconds.

## 7. Fast Existence Checks on Slow Virtual Filesystems (OneDrive)
- **Insight:** Virtual filesystems like OneDrive hook into sequential file operations. Calling `Path.exists()` sequentially 21,837 times in a loop takes over 10-15 minutes, causing training start stalls.
- **Solution:** Execute a single `os.walk` of the directory to collect all existing filenames in a Python `set`, and then perform constant-time set membership checks in memory, reducing startup overhead to under 0.2 seconds.

## 8. MLflow Model Serialization Formats
- **Insight:** MLflow's default model logging can attempt to trace PyTorch models with the `'pt2'` format, which requires a dummy input example.
- **Solution:** Specify `serialization_format="pickle"` to use standard object serialization when an input example is not readily available or when the model format is recurrent (LSTMs).

## 9. Early Stopping and Dropout Regularization for LSTMs
- **Insight:** Training high-capacity BiLSTMs (e.g. hidden size 256) on sequential ECG signals can quickly result in overfitting, where validation loss starts to increase while training loss continues to drop.
- **Solution:** Implementing standard early stopping (patience 7) based on validation loss, paired with robust dropout (LSTM: 0.4, FC: 0.5) and a Plateau learning rate scheduler (decay factor 0.1, patience 3) effectively prevents overfitting and stabilizes convergence.



