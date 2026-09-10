# Academic Thesis Notes: ECG Biomarker Representation Learning

## 1. Research Problem & Clinical Context
Standard deep learning on 12-lead ECG signals operates as a black box directly on raw voltage timeseries. While neural networks can discover high-dimensional temporal features, they risk learning spurious dataset artifacts (e.g., lead impedance differences, patient positioning shifts) and lack transparent alignment with clinical cardiology criteria established over a century of diagnostic practice.

This research establishes a domain-guided **Biomarker Encoder** that extracts 24 electrophysiological parameters defined in AHA/ACC/HRS guidelines, corrects for scale-space delineation bias using Continuous Wavelet Transforms (CWT), and trains multi-task neural representation models to project tabular clinical metrics into dense 32-dimensional latent representations.

---

## 2. Mathematical Formulation & Methods

### 2.1 Wavelet Scale-Space Delineation
Given a clean 12-lead signal $s(t)$, multi-scale Continuous Wavelet Transform is computed using the complex Morlet / Mexican hat wavelet family:
$$W_s(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} s(t) \psi^*\left(\frac{t - b}{a}\right) dt$$
Scale parameters $a$ are tuned across conduction frequency bands ($10-30\,\text{Hz}$ for QRS complexes, $1-5\,\text{Hz}$ for P/T waves), providing exact boundary localization without dyadic discretization artifacts.

### 2.2 Multi-Task Loss Formulation
The network optimizes a dual objective:
$$\min_{\theta, \phi, \psi} \mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{train}}} \left[ \mathcal{L}_{\text{recon}}(x, g_\phi(f_\theta(x))) + \lambda \mathcal{L}_{\text{cls}}(y, h_\psi(f_\theta(x))) \right]$$
where $f_\theta: \mathbb{R}^{48} \to \mathbb{R}^{32}$ is the encoder, $g_\phi: \mathbb{R}^{32} \to \mathbb{R}^{24}$ is the reconstruction decoder, and $h_\psi: \mathbb{R}^{32} \to \mathbb{R}^5$ is the multi-label diagnostic projection head.

---

## 3. Verified Empirical Results

| Representation / Model | Parameters | Recon MSE | Downstream Macro F1 | Downstream ROC-AUC | Downstream PR-AUC |
|---|---|---|---|---|---|
| **Raw Clinical Features (CWT)** | — | — | 0.5738 | 0.8040 | 0.5570 |
| **Preprocessed Baseline (Imputed + Scaled)** | — | — | 0.5885 | 0.8158 | 0.5834 |
| **FT-Transformer Autoencoder** | 74,173 | **0.1286** | 0.6192 | 0.8575 | 0.6786 |
| **Attention MLP Autoencoder** | 205,789 | 0.1813 | 0.6332 | **0.8622** | 0.6876 |
| **Beta-VAE Autoencoder** | 108,829 | 0.4130 | **0.6347** | 0.8607 | **0.6918** |

---

## 4. Key Scientific Findings
1. **Denoising and Representation Duality**: The multi-task latent space functions as an effective nonlinear denoiser. Projecting 24 clinical features into a regularized 32-D space increases downstream Macro F1 by **+6.09%** over raw features, demonstrating that latent manifold learning extracts higher-order diagnostic interactions than raw linear feature spaces.
2. **Clinical Delineation Independence**: While machine learning models are resilient to systematic translation offsets in features, physiological correctness (QRS duration ~106 ms, PR interval ~148 ms) is required for clinical acceptance and hybrid multi-modal interpretability.
3. **Disentanglement without Collapse**: All 32 latent dimensions remained active across all architectures (0 collapsed dimensions), with Beta-VAE providing a continuous probabilistic manifold and Attention MLP providing maximal diagnostic variance.
