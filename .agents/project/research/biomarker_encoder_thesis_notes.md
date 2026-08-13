# Biomarker Encoder Thesis Notes

This file integrates the Biomarker Encoder research findings into the project research documentation.

## Chapter 3: Methodology

### 3.4 Biomarker Encoder Architecture & Pipeline

We develop a structured tabular feature autoencoding framework to learn compact, 32-dimensional patient-level representations from continuous demographic, HRV, and morphology biomarkers.

#### Preprocessing & Imputation Masking
- **Median Imputation**: Features with missing entries are filled using train-split median statistics.
- **Binary Missingness Masking**: A helper mask vector $M \in \{0, 1\}^N$ is constructed to explicitly represent feature availability (1 if present, 0 if missing).
- **Concatenated Joint Features**: The input to the autoencoders is a joint vector $X_{\text{combined}} = [X_{\text{scaled}} \,\|\, M] \in \mathbb{R}^{2N}$.
- **Outlier Clamping**: Extreme values are clipped to the 1st and 99th percentiles to avoid loss divergence.

#### Expanded 60-Biomarker Clinical Set
To improve the clinical representation strength, we upgraded the feature extraction pipeline from 50 features ($2N=100$) to 60 base biomarkers ($2N=1212$ when fully expanded across 12 leads and paired with missingness masks, plus global clinical indices):
* **New Lead-Specific features**: P-wave polarity, QRS amplitude ($R + |S|$), Q-wave amplitude, J-point amplitude, ST-segment area (integration), T-wave polarity, T-wave peak-to-onset time, secondary $R'$ and $S'$ amplitudes, and ST-T slope relationship.
* **Global Precordial & Multi-lead features**: Sokolow-Lyon voltage, Cornell voltage, Cornell voltage-duration product, frontal axes (P, QRS, T) and QRS-T angle, precordial progressions, and QRS voltage dispersion across all leads.
* **Global HRV updates**: Added $NN50$, $QTc_{\text{Framingham}}$, and $QTc_{\text{Hodges}}$.

#### Model Specifications
- **Attention MLP Autoencoder**: Combines dense multi-layer perceptron projections with residual connections and a **MultiheadAttention** layer acting over hidden units. The decoder predicts the original $N$-dimensional scaled inputs to prevent reconstructing the binary mask. A classification head projects the bottleneck $z \to 5$ target classes.
- **Beta-VAE**: Implements a Variational Autoencoder containing dense encoder layers mapping inputs to mean $\mu$ and variance $\sigma^2$. Computes reconstruction loss alongside KL divergence scaled by $\beta=1.0$.
- **FT-Transformer**: Implements Feature Tokenization to project each continuous feature into a $32$-dimensional space. Token sequences are concatenated with a learnable `[CLS]` token and passed through a Transformer Encoder block to extract the latent embedding.

---

## Chapter 4: Results & Discussion

Quantitative benchmarking was performed on the PTB-XL dataset splits comparing the old (256 dimensions) and new (606 dimensions) biomarker setups.

### Feature Set Comparison Benchmarks

The models were trained for 5 epochs to compare the representation capability of the old versus the new feature sets:

| Model Type | Feature Set | Input Size | Reconstruction MSE | Reconstruction MAE |
| --- | :---: | :---: | :---: | :---: |
| **Attention MLP (Old)** | Old | 256 | 0.929552 | 0.705530 |
| **Attention MLP (New)** | New | 606 | **0.888992** | **0.684281** |
| **Beta-VAE (Old)** | Old | 256 | 0.974626 | 0.726040 |
| **Beta-VAE (New)** | New | 606 | **0.959030** | **0.711541** |
| **FT-Transformer (Old)** | Old | 256 | 0.973266 | 0.725279 |
| **FT-Transformer (New)** | New | 606 | **0.958384** | **0.710791** |

### Key Findings
1. **Lower Reconstruction Error**: For all three models, training on the new clinical feature set yields lower reconstruction error (both MSE and MAE). This confirms that autoencoders successfully map complex clinical markers to low-dimensional representations.
2. **Attention MLP** yielded the lowest reconstruction error overall (MSE: 0.888992), showing that residual connections and multihead self-attention over dense projections are highly effective for tabular feature compression.
3. **Representation Stability**: The FT-Transformer autoencoder shows highly stable reconstruction performance across the expanded feature set, aligning with the project's recommendations.
