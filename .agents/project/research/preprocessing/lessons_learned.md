# Signal Preprocessing Lessons Learned

This document logs critical research insights, environment constraints, and pipeline optimizations discovered during the development of the Signal Preprocessing module.

---

## 1. Zero-Phase Filtering for Cardiac Signals
- **Insight:** Direct lowpass/highpass/bandpass filters introduce phase distortion, which shifts peak locations (e.g. shifts R-peaks slightly to the right). This interferes with high-precision morphological segmentations like the Pan-Tompkins QRS algorithm.
- **Solution:** Always apply filtering using zero-phase forward-backward filtering (`scipy.signal.filtfilt`). This applies the filter coefficients in both directions, completely canceling phase shift.

---

## 2. Wavelet Denoising over Simple Smoothing
- **Insight:** Lowpass filtering removes high-frequency muscle noise but can also flatten the high-frequency peaks of QRS complexes, changing R-peak amplitude and QRS shape.
- **Solution:** Discrete Wavelet Transform (DWT) decomposition (e.g., using `db4` wavelet at level 4) with soft thresholding allows selective noise removal while preserving the sharp peaks of R-waves.

---

## 3. Multi-Label Resampling via Power-Set Mapping
- **Insight:** Standard class balancing algorithms (like SMOTE, ADASYN, and ENN) in `imbalanced-learn` do not support multi-hot label targets.
- **Solution:** Convert the multi-hot label matrix into unique integer class IDs using a label power-set mapping before resampling. Apply SMOTE-ENN on the resampled 2D flattened signals, and map the integer labels back to the original multi-hot label arrays.
