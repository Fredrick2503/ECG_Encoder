# Signal Preprocessing Thesis Methodology & Implementation Notes

These notes summarize the methodology and implementation sections of the Signal Preprocessing research for inclusion in the thesis document.

---

## Chapter 3: Methodology

### 3.2 Signal Preprocessing Pipeline
To guarantee reproducibility and modularity, the preprocessing pipeline is built strictly using open-source scientific computing libraries (`NumPy`, `SciPy`, `PyWavelets`, `scikit-learn`, `imbalanced-learn`).

```
+------------------+     +------------------+     +------------------+
|    Validation    | --> |    Filtering     | --> |   Segmentation   |
| (Flatline/NaNs)  |     | (Butterworth/DWT)|     | (Pan-Tompkins)   |
+------------------+     +------------------+     +------------------+
                                                           |
                                                           v
+------------------+     +------------------+     +------------------+
|  SMOTE-ENN Train | <-- |  DBSCAN Outliers | <-- |  Normalization   |
|  (Data Balance)  |     |  (Feature Dist.) |     |  (Z-Score/Robust)|
+------------------+     +------------------+     +------------------+
```

#### 3.2.1 Filtering Algorithms
- **Baseline Wander Removal:** A zero-phase 4th-order Butterworth high-pass filter ($f_c = 0.5\text{ Hz}$) removes low-frequency baseline drifts without introducing phase delay.
- **Powerline Interference Removal:** A zero-phase IIR Notch filter ($f_n = 60\text{ Hz}$, $Q = 30$) rejects electrical grid artifacts.
- **Wavelet Denoising:** A Discrete Wavelet Transform (DWT) using the Daubechies 4 (`db4`) wavelet decomposes signals into 4 levels. A soft threshold based on the median absolute deviation (MAD) of detail coefficients removes high-frequency muscle noise while maintaining clean QRS peaks.

#### 3.2.2 Normalization Techniques
- **Z-Score Normalization:** Standardizes leads independently to zero mean and unit variance.
- **Min-Max Scaling:** Linearly scales signals to a fixed range $[0, 1]$, standard for morphology-based networks.
- **Robust Scaling:** Uses median and interquartile range (IQR) to scale signals, preventing transient noise spikes from scaling down valid ECG wave complexes.

#### 3.2.3 QRS Beat Detection & Segmentation
- Heartbeats are segmented using a Python implementation of the Pan-Tompkins algorithm:
  1. Bandpass filter ($5\text{--}15\text{ Hz}$) to isolate the QRS energy band.
  2. Five-point derivative to capture slope.
  3. Squaring function to enhance QRS complexes.
  4. Moving integration window ($150\text{ ms}$).
  5. Peak detection on the integrated signal, followed by a local maximum search on raw waveforms to locate exact R-peaks.
- Heartbeats are segmented using a window of 100 samples before the peak ($200\text{ ms}$) and 150 samples after ($300\text{ ms}$), creating standardized beat segments of shape `(12, 250)` at 500 Hz.

#### 3.2.4 Outlier and Balance Processing
- **DBSCAN Clustering:** Extract mean, standard deviation, min, max, and power spectrum energy features per lead. Standardize the descriptors and run DBSCAN clustering ($\epsilon = 3.0, N_{\text{min}} = 5$). Points marked as outlier index $-1$ are automatically excluded from training.
- **Class Balancing:** Address class imbalance by converting multi-hot target labels to unique single-class IDs using a label power-set mapping, applying SMOTE-ENN, and reconstructing resampled multi-hot targets and signals.
