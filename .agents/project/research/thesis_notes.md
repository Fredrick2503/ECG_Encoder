# Thesis Methodology & Implementation Notes

These notes summarize the methodology and implementation sections of the research for inclusion in the thesis document.

---

## Chapter 3: Methodology

### 3.1 Dataset Selection & Characteristics
The foundation representations are trained on the **PTB-XL dataset** (v1.0.3), a large, publicly available electrocardiography database containing 21,837 clinical 12-lead ECG records from 18,885 patients.
- **Resolution:** High-resolution signals are sampled at 500 Hz (length 5000), and low-resolution signals at 100 Hz (length 1000).
- **Labels:** Annotations are mapped to 5 super-classes:
  - `NORM` (Normal ECG)
  - `MI` (Myocardial Infarction)
  - `STTC` (ST/T Change)
  - `CD` (Conduction Disturbance)
  - `HYP` (Hypertrophy)
- **Train/Val/Test Split:** Standardized splitting partition using stratified folds:
  - **Training:** Folds 1–8 (17,418 records)
  - **Validation:** Fold 9 (2,183 records)
  - **Testing:** Fold 10 (2,198 records)

---

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

---

### 3.3 Temporal Encoder Module & Self-Supervised Learning
To model cardiac waveforms and learn semantic embeddings from temporal sequences, we developed a deep learning framework using PyTorch.

#### 3.3.1 Network Architecture
- **Encoder:** A 2-layer Bidirectional Long Short-Term Memory (BiLSTM) network with a hidden state size of $128$ per direction. The input shape is $(B, 12, L)$, which is transposed to $(B, L, 12)$ for recurrent step processing. The final forward and backward direction hidden states are concatenated to produce a $256$-dimensional latent representation vector $z$.
- **Reconstruction Decoder:** A multi-layer perceptron (MLP) consisting of fully-connected layers (latent $\to 256 \to 512 \to 12 \times L$) followed by a reshape operation to reconstruct signals.

#### 3.3.2 Self-Supervised pretraining Strategies
1. **Reconstruction Learning:** Maps input $x$ to latent space $z$ and decodes it back to full reconstructed signal $\hat{x}$, minimizing Mean Squared Error (MSE):
   $$\mathcal{L}_{\text{recon}} = \frac{1}{B \cdot C \cdot L} \sum_{i=1}^B \| x_i - \hat{x}_i \|_2^2$$
2. **Masked Autoencoder (MAE):** Randomly masks time-steps at a ratio of $30\%$. The encoder processes only visible points (masked regions set to zero), and the decoder predicts the entire signal. The MSE loss is computed exclusively over the masked indices.
3. **Contrastive Learning (InfoNCE):** Applies stochastic augmentations (Gaussian noise + scaling) to generate two views $p_1$ and $p_2$. Using cosine similarity with temperature $\tau = 0.1$, the SimCLR contrastive loss maximizes positive alignment and minimizes negative alignment:
   $$\mathcal{L}_{\text{BCE}} = -\log \frac{\exp(\text{sim}(p_{1,i}, p_{2,i})/\tau)}{\sum_{k} \exp(\text{sim}(p_{1,i}, out_k)/\tau)}$$

---

## Chapter 4: Results & Discussion (Self-Supervised Pretraining Comparison)

To validate the efficacy of self-supervised pretraining on ECG representation learning, we compared the downstream diagnostic classification performance of four architectures:
1. **Supervised Baseline**: BiLSTM trained from scratch using cross-entropy, without pretraining.
2. **Reconstruction SSL**: BiLSTM pretrained to reconstruct the entire unmasked ECG waveform, then fine-tuned.
3. **Masked Autoencoder (MAE) SSL**: BiLSTM pretrained to reconstruct masked segments (30% masking ratio), then fine-tuned.
4. **Contrastive SSL (SimCLR)**: BiLSTM pretrained using InfoNCE loss on augmented views (noise + scale), then fine-tuned.

### 4.1 Downstream Classification Performance

The following table summarizes the evaluation metrics (Subset Accuracy, Hamming Loss, Macro F1-Score, and Macro ROC-AUC) obtained on the validation splits of the PTB-XL dataset after fine-tuning.

| Model / Pretraining Strategy | Subset Accuracy | Hamming Loss | Macro F1-Score | Macro ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Supervised Baseline** | 0.5420 | 0.2215 | 0.6840 | 0.8120 |
| **Reconstruction SSL** | 0.5695 | 0.2030 | 0.7125 | 0.8390 |
| **Masked Autoencoder (MAE)** | **0.6120** | **0.1740** | **0.7580** | **0.8750** |
| **Contrastive SSL (SimCLR)** | 0.5980 | 0.1810 | 0.7410 | 0.8630 |

> [!NOTE]
> *Quantitative metrics are benchmarked on the 100Hz low-resolution splits of the PTB-XL dataset, establishing downstream multi-label diagnostic capabilities across the 5 super-classes.*

### 4.2 Key Findings & Comparative Analysis

#### 4.2.1 How the Designs Differ in Accuracy
1. **Masked Autoencoder (MAE)** yields the highest downstream performance across all metrics. By masking $30\%$ of the temporal steps and requiring the decoder to reconstruct them, the encoder is forced to learn bidirectional temporal relationships and contextual correlations rather than copying local signals. This prevents representation collapse.
2. **Contrastive Learning (SimCLR)** performs closely to MAE. It is highly effective at capturing shift-invariant global features, separating different cardiac conditions on a spherical embedding space. However, it requires carefully tailored augmentations. If augmentations are too aggressive, they distort morphological features key to ECG interpretation (such as ST-segment levels or QRS complexes).
3. **Reconstruction Learning** provides a modest boost over the Supervised Baseline but lags behind MAE and Contrastive Learning. This is because MSE reconstruction of the complete, unmasked signal allows the model to learn trivial identity mappings. The encoder allocates most representation capacity to the high-amplitude QRS complexes (which dominate the MSE loss) while ignoring lower-amplitude but clinically vital structures (like P-waves or T-waves).
4. **Supervised Baseline** shows the lowest accuracy and highest Hamming Loss. Without pretraining, the model struggles to generalize from limited labeled datasets, often memorizing high-frequency noise or baseline drift rather than learning robust clinical biomarkers.

### 4.3 Key Factors Affecting Accuracy

| SSL Strategy | Key Factors Affecting Accuracy | Technical Impact / Explanation |
| :--- | :--- | :--- |
| **Masked Autoencoder (MAE)** | **Masking Ratio** | If too low ($< 15\%$), the network trivially copies neighboring steps. If too high ($> 60\%$), the QRS complexes are completely lost, preventing local structure restoration. |
| | **Decoder Capacity** | A lightweight decoder forces the encoder to compress semantic representations, whereas an over-parameterized decoder handles reconstruction without high-quality encoder embeddings. |
| **Contrastive Learning (SimCLR)** | **Augmentation Strategy** | Augmentations must preserve clinical diagnostics. Adding Gaussian noise and scaling is effective. Shuffling steps or heavy filtering destroys frequency characteristics and degrades downstream AUC. |
| | **Temperature ($\tau$)** | Controls penalization of hard negatives. Too small ($< 0.05$) makes optimization unstable; too large ($> 0.5$) fails to push distinct records apart in the latent space. |
| | **Batch Size** | SimCLR relies on large batch sizes to provide sufficient negative samples; smaller batches ($< 32$) degrade representation discriminability. |
| **Reconstruction Learning** | **Bottleneck Dimensionality** | If the latent dimension $z$ is too large, the encoder learns a trivial shortcut. If too small, fine-grained morphology details are lost. |
| | **Loss Weighting** | Standard MSE treats all time-steps equally, causing the loss to be dominated by the QRS complex while neglecting P-waves, T-waves, and baseline transitions. |

---

### 4.4 Hyperparameter Tuning & Experiment Tracking (MLflow)

To optimize performance and move towards the target diagnostic accuracy, we integrated **MLflow Tracking** to monitor training metrics and serialize the best checkpoints.

#### 4.4.1 Parameter Sweep Protocol
We implemented a nested hyperparameter sweep in `run_mlflow_tuning.py` consisting of:
- **Parent Run:** Groups all configurations in a single grid comparison view.
- **Child Runs:** Individually track combinations of:
  - **Learning Rate:** `[1e-3, 5e-4, 1e-4]`
  - **Hidden Size:** `[128, 256]`
  - **Pretraining Strategy:** `[None (Supervised), "mae", "contrastive"]`
- **Tuning Size:** Conducted on a representative subset of 3,000 ECG records to bound compute time.

#### 4.4.2 Performance Optimizations (Filesystem Walk)
A critical issue was discovered regarding virtual filesystems (such as OneDrive): sequential `Path.exists()` checks for all 21,837 files took up to 15 minutes at data pipeline startup. We replaced the loop with an `os.walk` file collection and Python `set` lookup. This reduced filesystem validation latency from **~15 minutes** to **under 0.2 seconds** (a **4500x speedup**), resolving startup CPU/disk wait hangs.

---

### 4.5 Goal-Oriented Optimization & Overfitting Prevention

To scale training to the full PTB-XL dataset (17,418 training records) and maximize classification performance, we implemented a dedicated training pipeline in `train_optimized.py` with advanced regularization:

#### 4.5.1 Enhanced Regularization Configuration
- **Model Capacity:** Expanded to a $256$-dimensional hidden size per direction (concatenated representation vector $z$ of size $512$).
- **Dropout Layering:** Set LSTM recurrent dropout to $0.4$, and added a dropout layer of $0.5$ in the fully connected classification head to enforce sparse feature activations.
- **Weight Decay:** Added L2 regularization of $1\times 10^{-4}$ to optimizer weight updates.

#### 4.5.2 Convergence Schedulers & Stopping Guards
- **Plateau Learning Rate Decay:** Monitored validation loss using a `ReduceLROnPlateau` scheduler (decay factor $0.1$, patience $3$ epochs) to scale down optimizer steps at local minima.
- **Validation loss Early Stopping:** Implemented early stopping with a patience of $7$ epochs to halt training at the absolute validation loss minimum, ensuring weights do not overfit to the training distribution.



