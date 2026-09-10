# Biomarker Encoder Lessons Learned & Research Insights

This document captures key research insights, pitfalls encountered, algorithmic trade-offs, and design principles discovered during the development and clinical validation of the Biomarker Encoder subsystem.

---

## 1. Wavelet Delineation Methods: DWT vs. CWT
- **Observation**: NeuroKit2's default Discrete Wavelet Transform (`dwt`) delineator uses fixed dyadic dyadic scales and hard thresholding. On noisy or low-amplitude ECG leads, it placed QRS onset markers up to 30ms too early and offset markers up to 30ms too late. This artificially inflated the median QRS duration from the true ~106ms to ~170ms, triggering abnormal warnings on 83% of records.
- **Solution**: Continuous Wavelet Transform (`cwt`) evaluates continuous scale-space responses, providing smooth and robust boundary localization across noisy baselines. This brought median QRS to 105.8 ms and PR interval to 148.0 ms.
- **Key Lesson**: In biomedical signal processing, machine learning models can easily learn from systematically biased features, but **clinical validity and physician interpretability require physiological calibration at the extraction stage**.

---

## 2. Multi-Task Regularization vs. Unconstrained Autoencoders
- **Observation**: Training an autoencoder purely on feature reconstruction ($\text{MSE}$) creates latent spaces that capture high-variance non-diagnostic features (such as noise or baseline shifts) while neglecting subtle diagnostic indicators (such as minor ST elevations or subtle T-wave inversions).
- **Solution**: Multi-task joint optimization combining reconstruction MSE with diagnostic cross-entropy ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda \mathcal{L}_{\text{cls}}$) forces the latent space to structure its 32 dimensions around diagnostic manifolds.
- **Result**: Downstream classification Macro F1 jumped from **0.5885** (preprocessed features) to **0.6347** (Beta-VAE) and **0.6332** (Attention MLP), proving that multi-task latent embeddings filter noise and act as an effective representation denoiser.

---

## 3. Preserving Information in Clinical Missingness
- **Observation**: In clinical ECGs, missing features are often non-random (MNAR). For example, missing P-waves and PR intervals strongly correlate with Atrial Fibrillation or Flutter. Imputing with the median alone eliminates the diagnostic indicator of absent waves.
- **Solution**: Appending a 24-dimensional binary missingness indicator mask ($M \in \{0, 1\}^{24}$) alongside the normalized values preserves the structural topology of missingness for the neural network.

---

## 4. Architectural Trade-offs: Attention MLP vs. Beta-VAE vs. FT-Transformer
- **Attention MLP (205k params)**: Yields the highest latent variance (99.55) and highest single-model ROC-AUC (0.8622). Excellent at capturing complex non-linear feature interactions via self-attention.
- **Beta-VAE (108k params)**: Yields the highest downstream Macro F1 (0.6347) and PR-AUC (0.6918). The KL divergence constraint smooths the representation space, preventing overfitting on rare pathology combinations.
- **FT-Transformer (74k params)**: Yields the lowest reconstruction error (MSE = 0.1286) with zero redundant dimension pairs. Ideal when strict feature reconstructibility is required.
