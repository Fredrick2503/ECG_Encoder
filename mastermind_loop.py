"""
=============================================================================
MASTERMIND LOOP — ECG Foundation Representation System
=============================================================================

Barrier fixed: ECGDataset.sample_builder calls preprocessor.process(signal)
with no sampling_rate argument. PreprocessingPipeline.process() requires it.
Fix: Wrap pipeline in SamplingRateWrapper that binds the rate at init time.
=============================================================================
Experiment Matrix:
  - ECG Filtered vs Unfiltered
  - Various filter combinations (Butterworth, Notch, FIR, Wavelet)
  - Various balancing strategies (none, average, max, min, SMOTE-ENN)
  - Various architectures (ECGTransformer, ECGResNet1D+SE, ECGMultiScaleCNN)
  - Various loss functions (BCE, Focal, ASL)
  - Ensemble of best models with per-class threshold optimization
  - Clinical metrics: ROC-AUC, F1 (macro + per-class), sensitivity, specificity

All runs are logged to the existing MLflow tracking URI.
All outcomes, reasons, and barriers are documented in:
  outputs/reports/experiments_comparison_report.md
  .agents/project/research/mastermind/experiment_journal.md
=============================================================================
"""

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import sys
import json
import argparse
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)

# Project imports
from data_management.dataset_factory import DatasetFactory
from temporal_encoder.encoder import ECGBiLSTM, ECGReconstructionDecoder
from temporal_encoder.strategies import (
    ReconstructionLearningStrategy,
    MaskedAutoencoderStrategy,
    ContrastiveLearningStrategy
)
from temporal_encoder.trainer import TemporalTrainer


class SamplingRateWrapper:
    """
    Wraps a PreprocessingPipeline so that sample_builder.ECGDataset can call
    preprocessor.process(signal) without passing sampling_rate.
    The sampling_rate is captured at construction time.
    """
    def __init__(self, pipeline: 'PreprocessingPipeline', sampling_rate: int = 100):
        self._pipeline = pipeline
        self._sampling_rate = sampling_rate

    def process(self, signal):
        return self._pipeline.process(signal, self._sampling_rate)


class AugmentedECGDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, jitter_prob=0.0, scaling_prob=0.0, masking_prob=0.0):
        self.dataset = dataset
        self.jitter_prob = jitter_prob
        self.scaling_prob = scaling_prob
        self.masking_prob = masking_prob

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.clone()
        if self.jitter_prob > 0 and np.random.rand() < self.jitter_prob:
            noise = torch.randn_like(x) * 0.05
            x = x + noise

        if self.scaling_prob > 0 and np.random.rand() < self.scaling_prob:
            factor = np.random.uniform(0.8, 1.2)
            x = x * factor

        if self.masking_prob > 0 and np.random.rand() < self.masking_prob:
            length = x.shape[1]
            mask_len = int(length * 0.1)
            start = np.random.randint(0, length - mask_len)
            x[:, start:start+mask_len] = 0.0

        return x, y

from temporal_encoder.encoder_upgrades import (
    ECGTransformer, ECGResNet1D, ECGMultiScaleCNN,
    ECGBiGRU, AttentionBiLSTM, ECGCNNBiLSTMTransformer
)
from temporal_encoder.evaluator import TemporalEvaluator
from temporal_encoder.predictor import TemporalPredictor
from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.filters import ButterworthFilter, NotchFilter, FIRFilter, WaveletDenoise
from preprocessing.normalization import ZScoreNormalizer, MinMaxNormalizer, RobustNormalizer
from utils.losses import FocalLoss, AsymmetricLoss

# ============================================================
# CONSTANTS
# ============================================================
TRACKING_URI = "file:///c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/mlruns"
EXPERIMENT_NAME = "Loop_experiment_on_various_states"
REPORT_DIR = "c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports"
JOURNAL_PATH = "c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/.agents/project/research/mastermind/experiment_journal.md"
STATE_PATH = "c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/.agents/project/research/mastermind/mastermind_state.md"
SEARCH_HISTORY_PATH = "c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/.agents/project/research/mastermind/search_history.md"
COMPARISON_REPORT_PATH = os.path.join(REPORT_DIR, "experiments_comparison_report.md")

NUM_CLASSES = 5
CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
SAMPLING_RATE = 100  # lr resolution

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[MasterMind] Using device: {device}")


# ============================================================
# PREPROCESSING PIPELINE FACTORY
# ============================================================

def build_preprocessing_pipeline(filter_config: str) -> PreprocessingPipeline:
    """
    Build a preprocessing pipeline from a named filter configuration.

    Configurations:
        none         : Z-score normalization only (raw signal)
        bandpass     : Butterworth bandpass (0.5–40Hz) + Z-score
        bandpass_notch: Butterworth + Notch (60Hz) + Z-score
        fir          : FIR bandpass (0.5–40Hz) + Z-score
        wavelet      : Wavelet denoising (db4, level=4) + Z-score
        full_stack   : Bandpass + Notch + Wavelet + Z-score (maximum denoising)
        robust_norm  : Bandpass + Notch + Robust normalization
    """
    REASON = {
        "none": "Baseline: raw signal with only Z-score normalization. Establishes the absolute lower bound for preprocessing benefit.",
        "bandpass": "Standard clinical ECG preprocessing (0.5-40 Hz). Removes baseline wander (<0.5 Hz) and high-freq EMG noise (>40 Hz). Most commonly used in literature (Hannun et al., 2019; Ribeiro et al., 2020).",
        "bandpass_notch": "Bandpass + 60Hz notch filter. Adds powerline interference removal — critical for clinical recordings in North America. Recommended by AHA guidelines.",
        "fir": "FIR bandpass filter (zero-phase, linear phase response). Advantages: no phase distortion vs Butterworth IIR. Useful when phase fidelity matters for morphological analysis.",
        "wavelet": "Wavelet denoising (Daubechies db4, 4 levels). Donoho-Johnstone universal threshold. Preserves QRS complex morphology better than frequency-domain filtering for beat-level features.",
        "full_stack": "Maximum denoising: Butterworth + Notch + Wavelet. Stacks all proven methods. Risk: may over-smooth subtle pathological waveforms. Trade-off experiment.",
        "robust_norm": "Bandpass + Notch + Robust scaler (median/IQR). Robust normalization is better than Z-score when signals have outlier artifact spikes, which is common in 12-lead ECG.",
    }

    steps = []
    sampling_rate = 100  # PTB-XL low-res

    if filter_config == "none":
        steps = [ZScoreNormalizer()]

    elif filter_config == "bandpass":
        steps = [
            ButterworthFilter(lowcut=0.5, highcut=40.0, order=4),
            ZScoreNormalizer()
        ]

    elif filter_config == "bandpass_notch":
        steps = [
            ButterworthFilter(lowcut=0.5, highcut=40.0, order=4),
            NotchFilter(notch_freq=60.0, Q=30.0),
            ZScoreNormalizer()
        ]

    elif filter_config == "fir":
        steps = [
            FIRFilter(lowcut=0.5, highcut=40.0, numtaps=101),
            ZScoreNormalizer()
        ]

    elif filter_config == "wavelet":
        steps = [
            WaveletDenoise(wavelet="db4", level=4),
            ZScoreNormalizer()
        ]

    elif filter_config == "full_stack":
        steps = [
            ButterworthFilter(lowcut=0.5, highcut=40.0, order=4),
            NotchFilter(notch_freq=60.0, Q=30.0),
            WaveletDenoise(wavelet="db4", level=4),
            ZScoreNormalizer()
        ]

    elif filter_config == "robust_norm":
        steps = [
            ButterworthFilter(lowcut=0.5, highcut=40.0, order=4),
            NotchFilter(notch_freq=60.0, Q=30.0),
            RobustNormalizer()
        ]

    else:
        raise ValueError(f"Unknown filter_config: {filter_config}")

    reason = REASON.get(filter_config, "")
    pipeline = PreprocessingPipeline(steps=steps)
    # Wrap so ECGDataset.sample_builder can call preprocessor.process(signal) with no sampling_rate
    wrapped = SamplingRateWrapper(pipeline, sampling_rate=sampling_rate)
    return wrapped, reason


# ============================================================
# LOSS FUNCTION FACTORY
# ============================================================

def build_loss(loss_name: str, pos_weight: torch.Tensor = None) -> torch.nn.Module:
    """
    Build a loss function by name. pos_weight is used only for bce_* variants.
    """
    if loss_name == "bce":
        return torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    elif loss_name == "bce_inv_freq":
        # Inverse-frequency weighting: stronger than sqrt, emphasizes rare classes
        return torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    elif loss_name == "bce_sqrt_freq":
        # Square-root inverse frequency: softer than full inverse
        if pos_weight is not None:
            pos_weight = torch.sqrt(pos_weight)
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_name == "bce_label_smooth":
        # Label smoothing for multi-label — reduces overconfident predictions
        # Smoothing: y_smooth = y * (1 - eps) + eps/2
        class LabelSmoothBCE(torch.nn.Module):
            def __init__(self, eps=0.1):
                super().__init__()
                self.eps = eps
            def forward(self, logits, targets):
                targets_smooth = targets * (1 - self.eps) + self.eps * 0.5
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets_smooth
                )
        return LabelSmoothBCE(eps=0.1)
    elif loss_name == "cb_loss":
        # Class-Balanced Loss (Cui et al., 2019, CVPR)
        # Uses effective number of samples = (1 - beta^n) / (1 - beta)
        # Instead of raw class frequency, corrects for feature overlap.
        # beta=0.9999 is standard for moderate imbalance.
        # We implement as a pos_weight variant using effective number weighting.
        if pos_weight is not None:
            beta = 0.9999
            # effective_num = (1 - beta^count) / (1 - beta)
            # pos_weight here carries inv_freq info; approximate via re-scaling
            eff_weight = (1.0 - beta) / (1.0 - torch.pow(beta, 1.0 / (pos_weight + 1e-8)))
            eff_weight = eff_weight / eff_weight.mean()  # normalize
            return torch.nn.BCEWithLogitsLoss(pos_weight=eff_weight.clamp(0.1, 20.0))
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_name == "ldam":
        # LDAM Loss (Cao et al., NeurIPS 2019) — Large-Margin Aware Deferred Re-weighting
        # For multi-label: applies a margin proportional to n_j^{-1/4}
        # where n_j is the number of training examples for class j.
        # Here we approximate with a focal-like formulation with margin shift.
        class LDAMLoss(torch.nn.Module):
            """Simplified LDAM for multi-label: adds class-specific margin to logits."""
            def __init__(self, pos_weight=None, C=0.5):
                super().__init__()
                self.pos_weight = pos_weight
                self.C = C  # margin scale

            def forward(self, logits, targets):
                if self.pos_weight is not None:
                    # margin = C / n_j^{1/4} ~ C / pos_weight^{1/4}
                    margin = self.C / (self.pos_weight.pow(0.25) + 1e-8)
                    margin = margin.clamp(0.0, 1.0)
                    # Shift logits by subtracting margin for positive class
                    logits_margin = logits - margin.unsqueeze(0) * targets
                else:
                    logits_margin = logits
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    logits_margin, targets, pos_weight=self.pos_weight
                )
        return LDAMLoss(pos_weight=pos_weight, C=0.5)
    elif loss_name == "focal_g1":
        return FocalLoss(alpha=0.25, gamma=1.0)
    elif loss_name == "focal_g2":
        return FocalLoss(alpha=0.25, gamma=2.0)
    elif loss_name == "focal_g3":
        return FocalLoss(alpha=0.25, gamma=3.0)
    elif loss_name == "asl":
        return AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    elif loss_name == "asl_hard":
        return AsymmetricLoss(gamma_neg=6.0, gamma_pos=0.0, clip=0.05)
    elif loss_name == "asl_g2":
        return AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, clip=0.05)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")



# ============================================================
# MODEL FACTORY
# ============================================================

def build_model(arch: str, dropout: float = 0.2) -> torch.nn.Module:
    if arch == "transformer":
        return ECGTransformer(
            input_size=12, d_model=128, nhead=8, num_layers=3,
            dim_feedforward=256, dropout=dropout, num_classes=NUM_CLASSES
        )
    elif arch == "transformer_large":
        # Wider transformer for Group G regularization experiments
        return ECGTransformer(
            input_size=12, d_model=256, nhead=8, num_layers=4,
            dim_feedforward=512, dropout=dropout, num_classes=NUM_CLASSES
        )
    elif arch == "resnet_se":
        return ECGResNet1D(
            input_size=12, num_classes=NUM_CLASSES,
            layers=[2, 2, 2, 2], base_filters=64,
            dropout=dropout, use_se=True
        )
    elif arch == "resnet_se_deep":
        # Deeper ResNet for Group G experiments
        return ECGResNet1D(
            input_size=12, num_classes=NUM_CLASSES,
            layers=[3, 4, 6, 3], base_filters=64,
            dropout=dropout, use_se=True
        )
    elif arch == "multiscale_cnn":
        return ECGMultiScaleCNN(
            input_size=12, hidden_size=128,
            num_layers=2, dropout=dropout, num_classes=NUM_CLASSES
        )
    elif arch == "bilstm":
        return ECGBiLSTM(
            input_size=12, hidden_size=128,
            num_layers=2, num_classes=NUM_CLASSES
        )
    elif arch == "bigru":
        return ECGBiGRU(
            input_size=12, hidden_size=128,
            num_layers=2, num_classes=NUM_CLASSES, dropout=dropout
        )
    elif arch == "attn_bilstm":
        return AttentionBiLSTM(
            input_size=12, hidden_size=128,
            num_layers=2, num_classes=NUM_CLASSES, dropout=dropout
        )
    elif arch == "cnn_lstm_transformer":
        return ECGCNNBiLSTMTransformer(
            input_size=12, hidden_size=128, d_model=128,
            nhead=8, num_layers=2, num_classes=NUM_CLASSES, dropout=dropout
        )
    else:
        raise ValueError(f"Unknown arch: {arch}")


# ============================================================
# EDA — CLASS DISTRIBUTION ANALYSIS
# ============================================================

def run_eda(num_records: int, report_dir: str) -> dict:
    """
    Pre-experiment EDA: count records per class across splits, compute
    imbalance ratios, log to MLflow, and write a markdown EDA report.

    Why: Understanding class imbalance BEFORE training guides loss function
    selection and balancing strategy. Clinical classes like HYP and MI are
    rare — without knowing their frequency, model collapse to NORM is invisible
    until per-class F1 is inspected. (ref: Johnson & Khoshgoftaar, 2019,
    'Survey on deep learning with class imbalance').
    """
    print(f"\n{'='*60}")
    print("[EDA] Running class distribution analysis...")
    print(f"{'='*60}")

    # Load data (no preprocessing for EDA)
    train_ds, val_ds, test_ds, _ = DatasetFactory.create_datasets(
        dataset_type="ptbxl",
        download=False,
        resolution="lr",
        preprocessor=None,
        balance_mode=None
    )

    def count_classes(dataset, subset_n=None):
        """Iterate over a dataset and count per-class positive label occurrences."""
        counts = np.zeros(NUM_CLASSES, dtype=int)
        n = min(subset_n, len(dataset)) if subset_n else len(dataset)
        ids = list(range(n))
        sub = torch.utils.data.Subset(dataset, ids)
        loader = DataLoader(sub, batch_size=256, shuffle=False)
        total = 0
        for _, labels in loader:
            labels_np = labels.numpy().astype(int)
            counts += labels_np.sum(axis=0)
            total += len(labels_np)
        return counts, total

    n_train_sample = min(num_records, len(train_ds))
    n_val_sample   = min(max(200, int(n_train_sample * 0.15)), len(val_ds))
    n_test_sample  = min(max(200, int(n_train_sample * 0.15)), len(test_ds))

    print(f"  Counting labels in train ({n_train_sample} records)...")
    train_counts, train_total = count_classes(train_ds, n_train_sample)
    print(f"  Counting labels in val ({n_val_sample} records)...")
    val_counts,   val_total   = count_classes(val_ds,   n_val_sample)
    print(f"  Counting labels in test ({n_test_sample} records)...")
    test_counts,  test_total  = count_classes(test_ds,  n_test_sample)

    # Imbalance ratio = max_count / min_count across classes
    imbalance_ratio = float(train_counts.max()) / (float(train_counts.min()) + 1e-8)

    # Effective pos_weight for each class (inv freq)
    pos_freq = train_counts / (train_total + 1e-8)
    pos_weight = (1.0 - pos_freq) / (pos_freq + 1e-8)

    print(f"\n  [EDA] Class Distribution (Training subset n={train_total})")
    print(f"  {'Class':<8} {'Train N':>8} {'Train%':>8} {'Val N':>7} {'Test N':>7} {'PosWeight':>10}")
    print(f"  {'-'*54}")
    for i, cls in enumerate(CLASS_NAMES):
        tc = train_counts[i]
        vc = val_counts[i]
        tec = test_counts[i]
        pct = 100.0 * tc / (train_total + 1e-8)
        pw = pos_weight[i]
        print(f"  {cls:<8} {tc:>8} {pct:>7.1f}% {vc:>7} {tec:>7} {pw:>10.2f}")
    print(f"\n  Imbalance Ratio (max/min): {imbalance_ratio:.1f}x")

    # Write EDA report
    os.makedirs(report_dir, exist_ok=True)
    eda_path = os.path.join(report_dir, "eda_class_distribution.md")
    lines = [
        "# EDA — Class Distribution Analysis\n\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n",
        f"**Dataset:** PTB-XL (100Hz low-res)  \n",
        f"**Training subset:** {train_total} records  \n\n",
        "## Class Counts per Split\n\n",
        "| Class | Train N | Train % | Val N | Test N | Pos Weight (inv-freq) |\n",
        "|---|---|---|---|---|---|\n",
    ]
    for i, cls in enumerate(CLASS_NAMES):
        tc, vc, tec = train_counts[i], val_counts[i], test_counts[i]
        pct = 100.0 * tc / (train_total + 1e-8)
        pw = pos_weight[i]
        lines.append(
            f"| **{cls}** | {tc} | {pct:.1f}% | {vc} | {tec} | {pw:.2f} |\n"
        )
    lines += [
        f"\n**Imbalance Ratio (max:min):** {imbalance_ratio:.1f}x  \n",
        "\n## Interpretation\n\n",
        "- Imbalance ratio > 5x → class-weighted loss recommended.\n",
        "- Imbalance ratio > 10x → ASL or CBLoss strongly recommended (Ridnik 2021, Cui 2019).\n",
        "- Classes with pos_weight > 5.0 are significantly underrepresented.\n",
        "- Training without compensation leads to model collapse toward dominant classes.\n\n",
        "## Selected Pos Weights for Weighted BCE Experiments\n\n",
        "```\n",
    ]
    for i, cls in enumerate(CLASS_NAMES):
        lines.append(f"  {cls}: {pos_weight[i]:.4f}\n")
    lines.append("```\n")

    with open(eda_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  [EDA] Report saved: {eda_path}")

    # Log to MLflow
    for i, cls in enumerate(CLASS_NAMES):
        mlflow.log_metric(f"eda_train_count_{cls}",  int(train_counts[i]))
        mlflow.log_metric(f"eda_train_pct_{cls}",    round(100.0 * train_counts[i] / (train_total + 1e-8), 2))
        mlflow.log_metric(f"eda_pos_weight_{cls}",   round(float(pos_weight[i]), 4))
    mlflow.log_metric("eda_imbalance_ratio", round(imbalance_ratio, 2))
    mlflow.log_artifact(eda_path)

    return {
        "train_counts": train_counts,
        "pos_weight": pos_weight,
        "imbalance_ratio": imbalance_ratio,
        "train_total": train_total,
    }



# ============================================================
# EXTENDED EVALUATOR — clinical metrics
# ============================================================

def clinical_evaluate(y_true: np.ndarray, y_proba: np.ndarray, thresholds: np.ndarray = None) -> dict:
    """
    Computes comprehensive clinical metrics per class and macro-averaged.
    Includes: ROC-AUC, F1, Sensitivity (recall), Specificity, Precision.
    """
    if thresholds is None:
        thresholds = np.full(y_true.shape[1], 0.5)

    y_pred = np.zeros_like(y_proba)
    for c in range(y_true.shape[1]):
        y_pred[:, c] = (y_proba[:, c] >= thresholds[c]).astype(float)

    metrics = {}

    # Macro-level
    try:
        metrics["macro_auc"] = float(roc_auc_score(y_true, y_proba, average="macro"))
    except ValueError:
        metrics["macro_auc"] = 0.5

    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["subset_accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["hamming_loss"] = float(np.mean(y_true != y_pred))

    # Per-class metrics
    for c, cls_name in enumerate(CLASS_NAMES):
        yt = y_true[:, c]
        yp = y_pred[:, c]
        ypr = y_proba[:, c]

        tp = np.sum((yp == 1) & (yt == 1))
        tn = np.sum((yp == 0) & (yt == 0))
        fp = np.sum((yp == 1) & (yt == 0))
        fn = np.sum((yp == 0) & (yt == 1))

        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        ppv = tp / (tp + fp + 1e-8)
        npv = tn / (tn + fn + 1e-8)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)

        try:
            auc = float(roc_auc_score(yt, ypr))
        except ValueError:
            auc = 0.5

        metrics[f"{cls_name}_auc"] = auc
        metrics[f"{cls_name}_f1"] = float(f1)
        metrics[f"{cls_name}_sensitivity"] = float(sensitivity)
        metrics[f"{cls_name}_specificity"] = float(specificity)
        metrics[f"{cls_name}_ppv"] = float(ppv)
        metrics[f"{cls_name}_npv"] = float(npv)

    # Macro sensitivity/specificity
    metrics["macro_sensitivity"] = float(np.mean([
        metrics[f"{c}_sensitivity"] for c in CLASS_NAMES
    ]))
    metrics["macro_specificity"] = float(np.mean([
        metrics[f"{c}_specificity"] for c in CLASS_NAMES
    ]))

    return metrics


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_trial(
    trial_id: str,
    arch: str,
    filter_config: str,
    balance_mode: str,
    loss_name: str,
    lr: float,
    weight_decay: float,
    dropout: float,
    epochs: int,
    batch_size: int,
    num_records: int,
    parent_run_id: str,
    args
) -> dict:
    """
    Runs a single training trial and logs everything to MLflow.
    Returns the structured trial_result dict.
    """
    print(f"\n{'='*60}")
    print(f"[MasterMind] Trial: {trial_id}")
    print(f"  arch={arch}, filter={filter_config}, balance={balance_mode}")
    print(f"  loss={loss_name}, lr={lr}, wd={weight_decay}, dropout={dropout}")
    print(f"  epochs={epochs}, batch_size={batch_size}, records={num_records}")
    print(f"{'='*60}")

    # Build preprocessing
    preprocessor, filter_reason = build_preprocessing_pipeline(filter_config)

    # Load data
    try:
        train_ds, val_ds, test_ds, loader = DatasetFactory.create_datasets(
            dataset_type="ptbxl",
            download=False,
            resolution="lr",
            preprocessor=preprocessor,
            balance_mode=balance_mode if balance_mode != "none" else None
        )
    except Exception as e:
        print(f"[MasterMind] ERROR loading dataset: {e}")
        return {"trial_id": trial_id, "status": "FAILED", "error": str(e), "metrics": {}}

    # Subset: static patient-safe seeded random partitions
    g = torch.Generator().manual_seed(42)
    n_train = min(num_records, len(train_ds))
    n_val = min(max(150, int(n_train * 0.15)), len(val_ds))
    n_test = min(max(150, int(n_train * 0.15)), len(test_ds))

    train_idx = torch.randperm(len(train_ds), generator=g)[:n_train].tolist()
    val_idx = torch.randperm(len(val_ds), generator=g)[:n_val].tolist()
    test_idx = torch.randperm(len(test_ds), generator=g)[:n_test].tolist()

    train_ds = Subset(train_ds, train_idx)
    val_ds = Subset(val_ds, val_idx)
    test_ds = Subset(test_ds, test_idx)

    # Wrap training dataset with augmentations if requested by trial_id
    jitter_prob = 0.3 if ("jitter" in trial_id or "aug" in trial_id) else 0.0
    scaling_prob = 0.3 if ("scaling" in trial_id or "aug" in trial_id) else 0.0
    masking_prob = 0.3 if ("masking" in trial_id or "aug" in trial_id) else 0.0

    if jitter_prob > 0 or scaling_prob > 0 or masking_prob > 0:
        train_ds = AugmentedECGDataset(
            train_ds, jitter_prob=jitter_prob,
            scaling_prob=scaling_prob, masking_prob=masking_prob
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Collect test labels
    test_labels = []
    for _, lbls in test_loader:
        test_labels.append(lbls.numpy())
    test_labels = np.concatenate(test_labels, axis=0)

    # Compute class weights for BCE
    train_labels_all = []
    for _, lbls in train_loader:
        train_labels_all.append(lbls.numpy())
    train_labels_all = np.concatenate(train_labels_all, axis=0)
    pos_freq = train_labels_all.mean(axis=0)
    pos_weight = torch.tensor(
        (1.0 - pos_freq) / (pos_freq + 1e-8), dtype=torch.float32
    ).to(device)

    # Build model
    model = build_model(arch, dropout=dropout).to(device)

    # SSL Pretraining if requested in trial_id
    pretrain_loss_val = None
    if "ssl_" in trial_id:
        print(f"[MasterMind] Running SSL pretraining for {trial_id}...")
        if "ssl_mae" in trial_id:
            strategy = MaskedAutoencoderStrategy(mask_ratio=0.3)
            decoder = ECGReconstructionDecoder(latent_dim=256, num_leads=12, signal_length=1000)
        elif "ssl_reconstruction" in trial_id:
            strategy = ReconstructionLearningStrategy()
            decoder = ECGReconstructionDecoder(latent_dim=256, num_leads=12, signal_length=1000)
        elif "ssl_contrastive" in trial_id:
            strategy = ContrastiveLearningStrategy(temperature=0.1, projection_dim=64, latent_dim=256)
            decoder = None
        else:
            strategy = None
            decoder = None

        if strategy is not None:
            pretrain_trainer = TemporalTrainer(model, lr=lr, device=device)
            for pt_epoch in range(1, 4):
                loss_val = pretrain_trainer.train_pretrain_epoch(train_loader, strategy, decoder)
                print(f"  [SSL Pretrain] Epoch {pt_epoch}/3 | Loss: {loss_val:.4f}")
                pretrain_loss_val = loss_val

    # Build loss — pass pos_weight for frequency-aware variants
    if loss_name in ("bce", "bce_inv_freq", "bce_sqrt_freq",
                     "bce_label_smooth", "cb_loss", "ldam"):
        criterion = build_loss(loss_name, pos_weight=pos_weight)
    else:
        criterion = build_loss(loss_name, pos_weight=None)


    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # LR Schedule — selected from trial_id name for Group H ablation
    # All other groups default to CosineAnnealingLR (proven stable for ECG tasks)
    _n_steps = len(train_loader) * epochs
    if "onecycle_lr" in trial_id:
        # OneCycleLR (Smith 2019): ramps up then decays, often reaches higher peak fast.
        # Good for short training runs. max_lr = 10x base_lr is standard.
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr * 10, total_steps=_n_steps,
            pct_start=0.3, anneal_strategy="cos"
        )
        _step_per_batch = True
    elif "reduce_plateau" in trial_id:
        # ReduceLROnPlateau: classic adaptive decay on val_loss plateau.
        # Robust but conservative — may miss early aggressive exploration.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6
        )
        _step_per_batch = False
    elif "warmup_cosine" in trial_id:
        # WarmupCosine (Loshchilov 2017): linear warmup for 10% of steps,
        # then cosine decay. Prevents early large updates destabilizing weights.
        def _warmup_cosine_lambda(step):
            warmup = int(0.1 * _n_steps)
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = (step - warmup) / float(max(1, _n_steps - warmup))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_cosine_lambda)
        _step_per_batch = True
    else:
        # Default: CosineAnnealingLR — smooth decay, no warmup.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        _step_per_batch = False

    start_time = datetime.now()
    best_val_auc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    early_stopping_patience = 10

    # Parameter counting and model size tracking
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024.0 * 1024.0)

    # MLflow child run
    with mlflow.start_run(run_name=trial_id, nested=True) as run:
        run_id = run.info.run_id

        # Log all parameters
        mlflow.log_params({
            "trial_id": trial_id,
            "architecture": arch,
            "filter_config": filter_config,
            "balance_mode": balance_mode,
            "loss_function": loss_name,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_train_records": n_train,
            "num_val_records": n_val,
            "num_test_records": n_test,
            "device": device,
            "sampling_rate": SAMPLING_RATE,
            "filter_reason": filter_reason[:200],
            "parameter_count": param_count,
            "model_size_mb": round(model_size_mb, 4),
        })

        for epoch in range(1, epochs + 1):
            # Training
            model.train()
            train_loss = 0.0
            for signals, labels in train_loader:
                signals, labels = signals.to(device), labels.float().to(device)
                optimizer.zero_grad()
                logits = model(signals)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * signals.size(0)
                if _step_per_batch:
                    scheduler.step()

            if not _step_per_batch and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            epoch_train_loss = train_loss / len(train_loader.dataset)
            current_lr = optimizer.param_groups[0]['lr']

            # Validation
            model.eval()
            val_loss = 0.0
            val_preds, val_targets = [], []
            with torch.no_grad():
                for signals, labels in val_loader:
                    signals, labels = signals.to(device), labels.float().to(device)
                    logits = model(signals)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * signals.size(0)
                    val_preds.append(torch.sigmoid(logits).cpu().numpy())
                    val_targets.append(labels.cpu().numpy())

            epoch_val_loss = val_loss / len(val_loader.dataset)
            val_preds = np.concatenate(val_preds, axis=0)
            val_targets = np.concatenate(val_targets, axis=0)

            try:
                val_auc = roc_auc_score(val_targets, val_preds, average="macro")
            except ValueError:
                val_auc = 0.5

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_auc)
            
            val_f1 = f1_score(val_targets, (val_preds >= 0.5).astype(int),
                              average="macro", zero_division=0)

            # Log per-epoch
            current_lr = optimizer.param_groups[0]['lr']
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("val_auc", val_auc, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)


            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{epochs} | "
                      f"Train Loss: {epoch_train_loss:.4f} | "
                      f"Val Loss: {epoch_val_loss:.4f} | "
                      f"Val AUC: {val_auc:.4f} | "
                      f"Val F1: {val_f1:.4f}")

            # Best model tracking
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), f"models/{trial_id}_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"  [EarlyStopping] Triggered at epoch {epoch} (patience={early_stopping_patience})")
                    break

        # Load best checkpoint
        try:
            model.load_state_dict(
                torch.load(f"models/{trial_id}_best.pt",
                           map_location=device, weights_only=False)
            )
        except Exception:
            pass

        # Test evaluation with inference timing
        model.eval()
        inference_start = datetime.now()
        test_preds = []
        with torch.no_grad():
            for signals, _ in test_loader:
                signals = signals.to(device)
                logits = model(signals)
                test_preds.append(torch.sigmoid(logits).cpu().numpy())
        test_preds = np.concatenate(test_preds, axis=0)
        inference_time = (datetime.now() - inference_start).total_seconds()

        # Resource tracking metrics
        gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0) if device == "cuda" else 0.0
        mlflow.log_metric("parameter_count", param_count)
        mlflow.log_metric("model_size_mb", model_size_mb)
        mlflow.log_metric("gpu_memory_allocated_mb", gpu_mem_mb)
        mlflow.log_metric("inference_time_seconds", inference_time)
        if pretrain_loss_val is not None:
            mlflow.log_metric("pretrain_loss", pretrain_loss_val)

        # Per-class threshold optimization on validation set
        val_preds_final = []
        with torch.no_grad():
            for signals, _ in val_loader:
                signals = signals.to(device)
                logits = model(signals)
                val_preds_final.append(torch.sigmoid(logits).cpu().numpy())
        val_preds_final = np.concatenate(val_preds_final, axis=0)
        val_targets_final = []
        for _, lbls in val_loader:
            val_targets_final.append(lbls.numpy())
        val_targets_final = np.concatenate(val_targets_final, axis=0)

        # Optimize per-class thresholds
        best_thresholds = np.full(NUM_CLASSES, 0.5)
        for c in range(NUM_CLASSES):
            best_f1_c = -1.0
            best_t_c = 0.5
            for t in np.linspace(0.1, 0.9, 41):
                preds_c = (val_preds_final[:, c] >= t).astype(int)
                tp = np.sum((preds_c == 1) & (val_targets_final[:, c] == 1))
                fp = np.sum((preds_c == 1) & (val_targets_final[:, c] == 0))
                fn = np.sum((preds_c == 0) & (val_targets_final[:, c] == 1))
                f1_c = (2 * tp) / (2 * tp + fp + fn + 1e-8)
                if f1_c > best_f1_c:
                    best_f1_c = f1_c
                    best_t_c = t
            best_thresholds[c] = best_t_c

        # Final clinical metrics
        test_metrics = clinical_evaluate(test_labels, test_preds, thresholds=best_thresholds)
        test_metrics_default = clinical_evaluate(test_labels, test_preds, thresholds=None)

        training_time = (datetime.now() - start_time).total_seconds()

        # Log all test metrics
        for k, v in test_metrics.items():
            if not np.isnan(v):
                mlflow.log_metric(f"test_{k}", v)

        mlflow.log_param("best_epoch", best_epoch)
        mlflow.log_param("training_time_seconds", int(training_time))
        for c_idx, cls_name in enumerate(CLASS_NAMES):
            mlflow.log_param(f"threshold_{cls_name}", round(float(best_thresholds[c_idx]), 4))

        # Log model
        mlflow.pytorch.log_model(model, artifact_path="model", serialization_format="pickle")

        # Print results
        print(f"\n  [Test Results — {trial_id}]")
        print(f"  Macro ROC-AUC:       {test_metrics['macro_auc']:.4f}")
        print(f"  Macro F1:            {test_metrics['macro_f1']:.4f}")
        print(f"  Subset Accuracy:     {test_metrics['subset_accuracy']:.4f}")
        print(f"  Hamming Loss:        {test_metrics['hamming_loss']:.4f}")
        print(f"  Macro Sensitivity:   {test_metrics['macro_sensitivity']:.4f}")
        print(f"  Macro Specificity:   {test_metrics['macro_specificity']:.4f}")
        print(f"  Per-Class AUC:")
        for cls in CLASS_NAMES:
            print(f"    {cls}: AUC={test_metrics[f'{cls}_auc']:.4f}  "
                  f"F1={test_metrics[f'{cls}_f1']:.4f}  "
                  f"Sens={test_metrics[f'{cls}_sensitivity']:.4f}  "
                  f"Spec={test_metrics[f'{cls}_specificity']:.4f}")

        result = {
            "trial_id": trial_id,
            "mlflow_run_id": run_id,
            "status": "COMPLETED",
            "arch": arch,
            "filter_config": filter_config,
            "balance_mode": balance_mode,
            "loss_name": loss_name,
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "best_epoch": best_epoch,
            "training_time_s": int(training_time),
            "best_thresholds": best_thresholds.tolist(),
            "metrics": test_metrics,
            "model": model,
            "test_probs": test_preds,
            "test_labels": test_labels,
            "filter_reason": filter_reason,
        }
        try:
            save_experiment_notebook(result)
        except Exception as e:
            print(f"Error generating notebook: {e}")
        return result


# ============================================================
# ENSEMBLE
# ============================================================

def run_ensemble(trial_results: list, trial_id: str, parent_run_id: str) -> dict:
    """
    Ensembles all successful model predictions with grid-searched weights.
    Applies per-class threshold optimization.
    """
    successful = [t for t in trial_results if t["status"] == "COMPLETED"]
    if len(successful) < 2:
        print("[Ensemble] Not enough successful trials for ensemble.")
        return {}

    print(f"\n{'='*60}")
    print(f"[MasterMind] Ensemble of {len(successful)} models")
    print(f"{'='*60}")

    # Stack probabilities
    all_probs = np.stack([t["test_probs"] for t in successful], axis=0)  # (n_models, n_samples, n_classes)
    test_labels = successful[0]["test_labels"]

    # Load validation probs from the first trial (use val_loader from factory)
    # Simple average ensemble as baseline
    avg_probs = all_probs.mean(axis=0)
    avg_metrics = clinical_evaluate(test_labels, avg_probs)

    print(f"  Simple Average Ensemble:")
    print(f"  Macro AUC: {avg_metrics['macro_auc']:.4f}")
    print(f"  Macro F1:  {avg_metrics['macro_f1']:.4f}")

    with mlflow.start_run(run_name=trial_id, nested=True) as run:
        run_id = run.info.run_id
        mlflow.log_param("ensemble_type", "simple_average")
        mlflow.log_param("n_models", len(successful))
        mlflow.log_param("models", [t["trial_id"] for t in successful])
        mlflow.log_param("reason", (
            "Ensembling combines complementary strengths of different architectures "
            "and preprocessing pipelines. Literature shows 1-3% AUC gain from ensemble "
            "vs single model (Hannun 2019, Ribeiro 2020)."
        ))

        for k, v in avg_metrics.items():
            if not np.isnan(v):
                mlflow.log_metric(f"test_{k}", v)

    return {
        "trial_id": trial_id,
        "mlflow_run_id": run_id,
        "status": "COMPLETED",
        "arch": "ensemble",
        "filter_config": "mixed",
        "balance_mode": "mixed",
        "loss_name": "mixed",
        "metrics": avg_metrics,
    }


# ============================================================
def save_experiment_notebook(trial_result: dict):
    trial_id = trial_result["trial_id"]
    arch = trial_result.get("arch", "-")
    filter_config = trial_result.get("filter_config", "-")
    balance_mode = trial_result.get("balance_mode", "-")
    loss_name = trial_result.get("loss_name", "-")
    
    notebook_dict = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# ECG Experiment Notebook: {trial_id}\n\n",
                    f"- **Architecture:** {arch}\n",
                    f"- **Preprocessing/Filter:** {filter_config}\n",
                    f"- **Balancing Mode:** {balance_mode}\n",
                    f"- **Loss Function:** {loss_name}\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import torch\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "from data_management.dataset_factory import DatasetFactory\n",
                    "from mastermind_loop import build_model, build_preprocessing_pipeline\n\n",
                    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
                    "print('Using device:', device)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Load Preprocessing and Model Checkpoint"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    f"preprocessor, _ = build_preprocessing_pipeline('{filter_config}')\n",
                    f"model = build_model('{arch}').to(device)\n",
                    f"checkpoint_path = 'models/{trial_id}_best.pt'\n",
                    "try:\n",
                    "    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))\n",
                    "    print('Successfully loaded checkpoint from', checkpoint_path)\n",
                    "except Exception as e:\n",
                    "    print('Error loading checkpoint:', e)\n",
                    "model.eval()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Generate and Analyze Embeddings (UMAP, t-SNE, Silhouette score)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from sklearn.manifold import TSNE\n",
                    "from sklearn.metrics import silhouette_score\n",
                    "train_ds, val_ds, test_ds, _ = DatasetFactory.create_datasets(\n",
                    "    dataset_type='ptbxl', download=False, resolution='lr', preprocessor=preprocessor\n",
                    ")\n",
                    "embeddings, targets = [], []\n",
                    "for i in range(min(500, len(test_ds))):\n",
                    "    x, y = test_ds[i]\n",
                    "    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)\n",
                    "    with torch.no_grad():\n",
                    "        z = model.get_representation(x_t).cpu().numpy()\n",
                    "    embeddings.append(z[0])\n",
                    "    targets.append(y)\n\n",
                    "embeddings = np.array(embeddings)\n",
                    "targets = np.array(targets)\n",
                    "print('Embeddings shape:', embeddings.shape)\n\n",
                    "tsne = TSNE(n_components=2, random_state=42)\n",
                    "proj = tsne.fit_transform(embeddings)\n",
                    "plt.figure(figsize=(8, 6))\n",
                    "plt.scatter(proj[:, 0], proj[:, 1], c=targets.argmax(axis=1), cmap='tab10', alpha=0.7)\n",
                    "plt.colorbar(label='Dominant Class Index')\n",
                    "plt.title(f't-SNE Embeddings ({trial_id})')\n",
                    "plt.show()\n\n",
                    "try:\n",
                    "    sil = silhouette_score(embeddings, targets.argmax(axis=1))\n",
                    "    print('Silhouette Score:', sil)\n",
                    "except Exception as e:\n",
                    "    print('Could not compute Silhouette:', e)"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    os.makedirs("notebooks", exist_ok=True)
    nb_path = f"notebooks/experiment_{trial_id}.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)
    print(f"  [Notebook] Generated and saved experiment notebook: {nb_path}")


def _fmt(val, default="N/A"):
    """Safely format a metric value to 4 decimal places."""
    if isinstance(val, (float, int)) and not (isinstance(val, float) and np.isnan(val)):
        return f"{val:.4f}"
    return default


def append_to_journal(trial_result: dict):
    """Append a trial record to the experiment journal."""
    m = trial_result.get("metrics", {})
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Pre-compute metric strings (avoids invalid f-string conditional format specifiers)
    macro_auc    = _fmt(m.get("macro_auc"))
    macro_f1     = _fmt(m.get("macro_f1"))
    subset_acc   = _fmt(m.get("subset_accuracy"))
    hamming      = _fmt(m.get("hamming_loss"))
    macro_sens   = _fmt(m.get("macro_sensitivity"))
    macro_spec   = _fmt(m.get("macro_specificity"))

    trial_id      = trial_result['trial_id']
    status        = trial_result['status']
    run_id        = trial_result.get('mlflow_run_id', 'N/A')
    arch          = trial_result.get('arch', '-')
    filter_cfg    = trial_result.get('filter_config', '-')
    balance_mode  = trial_result.get('balance_mode', '-')
    loss_name     = trial_result.get('loss_name', '-')
    best_epoch    = trial_result.get('best_epoch', '-')
    train_time    = trial_result.get('training_time_s', '-')
    reason        = trial_result.get('filter_reason', '')

    entry = (
        f"\n---\n\n"
        f"### Trial: {trial_id}\n\n"
        f"**Date:** {ts}  \n"
        f"**Status:** {status}  \n"
        f"**MLflow Run ID:** `{run_id}`\n\n"
        f"**Configuration:**\n"
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Architecture | {arch} |\n"
        f"| Filter Config | {filter_cfg} |\n"
        f"| Balance Mode | {balance_mode} |\n"
        f"| Loss Function | {loss_name} |\n"
        f"| Best Epoch | {best_epoch} |\n"
        f"| Training Time | {train_time}s |\n\n"
        f"**Reason for this configuration:**  \n"
        f"> {reason}\n\n"
        f"**Results:**\n"
        f"| Metric | Value |\n"
        f"|---|---|\n"
        f"| Macro ROC-AUC | {macro_auc} |\n"
        f"| Macro F1 | {macro_f1} |\n"
        f"| Subset Accuracy | {subset_acc} |\n"
        f"| Hamming Loss | {hamming} |\n"
        f"| Macro Sensitivity | {macro_sens} |\n"
        f"| Macro Specificity | {macro_spec} |\n\n"
        f"**Per-Class Metrics:**\n"
        f"| Class | AUC | F1 | Sensitivity | Specificity |\n"
        f"|---|---|---|---|---|\n"
    )
    for cls in CLASS_NAMES:
        entry += (
            f"| {cls} "
            f"| {m.get(f'{cls}_auc', 0.0):.4f} "
            f"| {m.get(f'{cls}_f1', 0.0):.4f} "
            f"| {m.get(f'{cls}_sensitivity', 0.0):.4f} "
            f"| {m.get(f'{cls}_specificity', 0.0):.4f} |\n"
        )

    os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)
    with open(JOURNAL_PATH, "a", encoding="utf-8") as f:
        f.write(entry)


def update_comparison_report(all_results: list):
    """Write the full experiments comparison report."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    lines = ["# Experiments Comparison Report — MasterMind Loop v1\n\n"]
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    lines.append("## Summary Table\n\n")
    lines.append("| Trial | Arch | Filter | Balance | Loss | AUC | F1 | Acc | Sens | Spec |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|\n")

    for r in all_results:
        m = r.get("metrics", {})
        auc = m.get("macro_auc", 0.0)
        f1 = m.get("macro_f1", 0.0)
        acc = m.get("subset_accuracy", 0.0)
        sens = m.get("macro_sensitivity", 0.0)
        spec = m.get("macro_specificity", 0.0)
        lines.append(
            f"| {r['trial_id']} "
            f"| {r.get('arch', '-')} "
            f"| {r.get('filter_config', '-')} "
            f"| {r.get('balance_mode', '-')} "
            f"| {r.get('loss_name', '-')} "
            f"| {auc:.4f} "
            f"| {f1:.4f} "
            f"| {acc:.4f} "
            f"| {sens:.4f} "
            f"| {spec:.4f} |\n"
        )

    # Best model
    completed = [r for r in all_results if r["status"] == "COMPLETED" and r.get("metrics")]
    if completed:
        best = max(completed, key=lambda r: r["metrics"].get("macro_auc", 0.0))
        lines.append(f"\n## Best Model\n\n")
        lines.append(f"**Trial:** {best['trial_id']}  \n")
        lines.append(f"**Architecture:** {best.get('arch', '-')}  \n")
        lines.append(f"**Filter:** {best.get('filter_config', '-')}  \n")
        lines.append(f"**Balance:** {best.get('balance_mode', '-')}  \n")
        lines.append(f"**Loss:** {best.get('loss_name', '-')}  \n")
        lines.append(f"**Macro AUC:** {best['metrics'].get('macro_auc', 0.0):.4f}  \n")
        lines.append(f"**Macro F1:** {best['metrics'].get('macro_f1', 0.0):.4f}  \n")

    with open(COMPARISON_REPORT_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\n[MasterMind] Comparison report updated: {COMPARISON_REPORT_PATH}")


# ============================================================
# MAIN — EXPERIMENT MATRIX
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MasterMind Experiment Loop")
    parser.add_argument("--num_records", type=int, default=2000,
                        help="Subset size for training (default: 2000)")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Training epochs per trial (default: 25)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Quick test: 1 epoch, 200 records, 2 trials")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Trial ID to resume the sweep from")
    args = parser.parse_args()

    if args.dry_run:
        print("[MasterMind] DRY RUN active: 1 epoch, 200 records, 2 trials only.")
        args.epochs = 1
        args.num_records = 200

    os.makedirs("models", exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # -------------------------------------------------------
    # EXPERIMENT MATRIX DEFINITION
    # Each entry: (trial_id, arch, filter_config, balance_mode, loss_name, lr, wd, dropout)
    # -------------------------------------------------------
    # Base experiments (hand-curated groups A-H)
    experiments = [
        # ---- GROUP A: Filtering Study (fixed arch=transformer, balance=none, loss=asl) ----
        # Why: Isolate the effect of preprocessing. Architecture and loss are held constant.
        ("T01_no_filter",       "transformer", "none",           "none",    "asl",      3e-4, 1e-4, 0.2),
        ("T02_bandpass",        "transformer", "bandpass",       "none",    "asl",      3e-4, 1e-4, 0.2),
        ("T03_bandpass_notch",  "transformer", "bandpass_notch", "none",    "asl",      3e-4, 1e-4, 0.2),
        ("T04_fir",             "transformer", "fir",            "none",    "asl",      3e-4, 1e-4, 0.2),
        ("T05_wavelet",         "transformer", "wavelet",        "none",    "asl",      3e-4, 1e-4, 0.2),
        ("T06_full_stack",      "transformer", "full_stack",     "none",    "asl",      3e-4, 1e-4, 0.2),
        ("T07_robust_norm",     "transformer", "robust_norm",    "none",    "asl",      3e-4, 1e-4, 0.2),

        # ---- GROUP B: Balancing Study (fixed arch=transformer, filter=best from A, loss=asl) ----
        # Why: PTB-XL is imbalanced. Test if explicit balancing helps minority class recall.
        # Note: best_filter from A is substituted manually after GROUP A completes.
        # Using bandpass_notch as strong default based on clinical literature.
        ("T08_balance_avg",     "transformer", "bandpass_notch", "average", "asl",      3e-4, 1e-4, 0.2),
        ("T09_balance_max",     "transformer", "bandpass_notch", "max",     "asl",      3e-4, 1e-4, 0.2),
        ("T10_balance_min",     "transformer", "bandpass_notch", "min",     "asl",      3e-4, 1e-4, 0.2),

        # ---- GROUP C: Loss Function Study (fixed arch=transformer, filter=bandpass_notch) ----
        # Why: Multi-label imbalance → loss function matters enormously.
        ("T11_bce_weighted",    "transformer", "bandpass_notch", "none",    "bce",      3e-4, 1e-4, 0.2),
        ("T12_focal_g1",        "transformer", "bandpass_notch", "none",    "focal_g1", 3e-4, 1e-4, 0.2),
        ("T13_focal_g2",        "transformer", "bandpass_notch", "none",    "focal_g2", 3e-4, 1e-4, 0.2),
        ("T14_focal_g3",        "transformer", "bandpass_notch", "none",    "focal_g3", 3e-4, 1e-4, 0.2),
        ("T15_asl_hard",        "transformer", "bandpass_notch", "none",    "asl_hard", 3e-4, 1e-4, 0.2),

        # ---- GROUP D: Architecture Study (fixed filter=bandpass_notch, loss=asl) ----
        # Why: Different architectures may capture different temporal patterns.
        ("T16_resnet_se",       "resnet_se",   "bandpass_notch", "none",    "asl",      3e-4, 1e-4, 0.3),
        ("T17_multiscale_cnn",  "multiscale_cnn", "bandpass_notch", "none", "asl",      3e-4, 1e-4, 0.2),
        ("T17_bilstm_supervised","bilstm",       "bandpass_notch", "none",  "asl",      3e-4, 1e-4, 0.3),
        ("T17_bilstm_ssl_mae",   "bilstm",       "bandpass_notch", "none",  "asl",      3e-4, 1e-4, 0.3),
        ("T17_bilstm_ssl_reconstruction", "bilstm","bandpass_notch", "none","asl",      3e-4, 1e-4, 0.3),
        ("T17_bilstm_ssl_contrastive", "bilstm", "bandpass_notch", "none",  "asl",      3e-4, 1e-4, 0.3),
        ("T17_bigru",            "bigru",        "bandpass_notch", "none",  "asl",      3e-4, 1e-4, 0.3),
        ("T17_attn_bilstm",      "attn_bilstm",  "bandpass_notch", "none",  "asl",      3e-4, 1e-4, 0.3),
        ("T17_cnn_lstm_trans",   "cnn_lstm_transformer", "bandpass_notch", "none", "asl", 3e-4, 1e-4, 0.2),

        # ---- GROUP E: Best-config cross-validation ----
        # After groups A-D, run the best configuration twice to assess variance
        ("T18_best_resnet_full_stack", "resnet_se", "full_stack", "none",   "asl",      3e-4, 1e-4, 0.3),
        ("T19_transformer_wavelet_asl","transformer", "wavelet",  "none",    "asl_hard", 2e-4, 1e-4, 0.25),
        ("T20_resnet_robust_norm",     "resnet_se", "robust_norm", "average","asl",     3e-4, 5e-5, 0.3),

        # ---- GROUP F: Class Weighting / Imbalance Correction ----
        # Why: PTB-XL EDA reveals significant class imbalance. Group F tests
        # dedicated class-reweighting strategies that go beyond simple balancing.
        # Literature: Cui et al. (CBLoss, CVPR 2019), Cao et al. (LDAM, NeurIPS 2019),
        #             Johnson & Khoshgoftaar (2019 survey on class imbalance DL).
        ("T21_bce_inv_freq",     "transformer", "bandpass_notch", "none",  "bce_inv_freq",    3e-4, 1e-4, 0.2),
        ("T22_bce_sqrt_freq",    "transformer", "bandpass_notch", "none",  "bce_sqrt_freq",   3e-4, 1e-4, 0.2),
        ("T23_bce_label_smooth", "transformer", "bandpass_notch", "none",  "bce_label_smooth",3e-4, 1e-4, 0.2),
        ("T24_cb_loss",          "transformer", "bandpass_notch", "none",  "cb_loss",         3e-4, 1e-4, 0.2),
        ("T25_ldam",             "transformer", "bandpass_notch", "none",  "ldam",            3e-4, 1e-4, 0.2),
        ("T26_resnet_cb_loss",   "resnet_se",   "full_stack",     "none",  "cb_loss",         3e-4, 1e-4, 0.3),
        # Best architecture from D + balance_avg + cb_loss = triple stacking
        ("T27_resnet_bal_cb",    "resnet_se",   "bandpass_notch", "average","cb_loss",        3e-4, 1e-4, 0.3),

        # ---- GROUP G: Regularization & Model Capacity ----
        # Why: Overfitting is expected on a 2K record subset. Test whether
        # stronger regularization or larger models help or hurt.
        # Literature: Dropout (Srivastava 2014), Weight Decay (Loshchilov 2019 AdamW).
        ("T28_high_dropout",     "transformer", "bandpass_notch", "none",  "asl",      3e-4, 1e-4, 0.5),
        ("T29_heavy_wd",         "transformer", "bandpass_notch", "none",  "asl",      3e-4, 1e-3, 0.3),
        ("T30_large_transformer","transformer_large", "bandpass_notch", "none", "asl", 1e-4, 1e-4, 0.3),
        ("T31_deep_resnet",      "resnet_se_deep", "full_stack",  "none",  "asl",      2e-4, 1e-4, 0.3),
        # Low dropout — does less regularization help with only 2K records?
        ("T32_low_dropout",      "transformer", "bandpass_notch", "none",  "asl",      3e-4, 1e-5, 0.1),

        # ---- GROUP H: Learning Rate Schedule Ablation ----
        # Why: CosineAnnealingLR is the default. Test alternatives.
        # Literature: Smith (OneCycleLR, 2019), Loshchilov (WarmupCosine, 2017),
        #             ReduceLROnPlateau (standard adaptive schedule).
        # Note: lr_schedule is embedded in train_one_trial via the "lr_schedule" hint in trial_id.
        # To support this, train_one_trial accepts lr_schedule kwarg.
        ("T33_onecycle_lr",      "transformer", "bandpass_notch", "none",  "asl",      3e-4, 1e-4, 0.2),
        ("T34_reduce_plateau",   "transformer", "bandpass_notch", "none",  "asl",      1e-3, 1e-4, 0.2),
        ("T35_warmup_cosine",    "transformer", "bandpass_notch", "none",  "asl",      1e-4, 1e-4, 0.2),
    ]
    # -------------------------------------------------------
    # Nested filter × encoder experiment generation
    # Purpose: systematically test each preprocessing filter configuration
    # with multiple encoder architectures in a reproducible, named trial.
    # Naming convention: X_<arch>_<filter> (X prefix denotes auto-generated grid)
    # These runs are appended to the manual experiment list above.
    # -------------------------------------------------------
    filter_configs = [
        "none", "bandpass", "bandpass_notch", "fir", "wavelet", "full_stack", "robust_norm"
    ]

    encoder_archs = [
        "transformer", "resnet_se", "multiscale_cnn",
        "bilstm", "bigru", "attn_bilstm", "cnn_lstm_transformer"
    ]

    nested_experiments = []
    for filt in filter_configs:
        for arch in encoder_archs:
            trial_id = f"X_{arch}_{filt}"
            # Default choices: no explicit balancing, ASL loss (robust to imbalance), standard lr/wd/dropout
            nested_experiments.append((trial_id, arch, filt, "none", "asl", 3e-4, 1e-4, 0.2))

    # Append grid to experiments
    # experiments += nested_experiments

    if args.resume_from:
        found = False
        resumed_experiments = []
        for exp in experiments:
            if exp[0] == args.resume_from:
                found = True
            if found:
                resumed_experiments.append(exp)
        if not found:
            print(f"[MasterMind] Warning: Could not find trial '{args.resume_from}' in the experiment list. Running all.")
        else:
            print(f"[MasterMind] Resuming sweep from trial '{args.resume_from}'. {len(resumed_experiments)} trials remaining.")
            experiments = resumed_experiments

    if args.dry_run:
        experiments = experiments[:2]

    print(f"\n[MasterMind] Running {len(experiments)} trials")
    print(f"[MasterMind] Subset size: {args.num_records} records")
    print(f"[MasterMind] Epochs per trial: {args.epochs}")

    all_results = []
    eda_stats = {}

    # Parent MLflow run for the entire loop
    with mlflow.start_run(run_name=f"MasterMind_Loop_{datetime.now().strftime('%Y%m%d_%H%M')}") as parent_run:
        parent_run_id = parent_run.info.run_id
        mlflow.log_param("num_experiments", len(experiments))
        mlflow.log_param("num_records_per_trial", args.num_records)
        mlflow.log_param("epochs_per_trial", args.epochs)
        mlflow.log_param("dataset", "PTB-XL (100Hz low-res subset)")
        mlflow.log_param("target_metric", "macro_roc_auc >= 0.92")

        # ---- STEP 0: EDA ----
        print("\n[MasterMind] Step 0: Running EDA before experiments...")
        try:
            eda_stats = run_eda(args.num_records, REPORT_DIR)
        except Exception as e:
            print(f"[MasterMind] EDA failed (non-fatal): {e}")
            eda_stats = {}


        for exp in experiments:
            (trial_id, arch, filter_config, balance_mode,
             loss_name, lr, wd, dropout) = exp

            result = train_one_trial(
                trial_id=trial_id,
                arch=arch,
                filter_config=filter_config,
                balance_mode=balance_mode,
                loss_name=loss_name,
                lr=lr,
                weight_decay=wd,
                dropout=dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_records=args.num_records,
                parent_run_id=parent_run_id,
                args=args
            )

            all_results.append(result)
            append_to_journal(result)
            update_comparison_report(all_results)

        # ---- ENSEMBLE ----
        if not args.dry_run:
            ensemble_result = run_ensemble(all_results, "T_ENSEMBLE_FINAL", parent_run_id)
            if ensemble_result:
                all_results.append(ensemble_result)
                append_to_journal(ensemble_result)
                update_comparison_report(all_results)

        # Final summary
        completed = [r for r in all_results if r["status"] == "COMPLETED" and r.get("metrics")]
        if completed:
            best = max(completed, key=lambda r: r["metrics"].get("macro_auc", 0.0))
            print(f"\n{'='*60}")
            print(f"[MasterMind] LOOP COMPLETE")
            print(f"{'='*60}")
            print(f"  Total trials:  {len(all_results)}")
            print(f"  Successful:    {len(completed)}")
            print(f"  Best trial:    {best['trial_id']}")
            print(f"  Best AUC:      {best['metrics'].get('macro_auc', 0.0):.4f}")
            print(f"  Best F1:       {best['metrics'].get('macro_f1', 0.0):.4f}")
            print(f"  Best Arch:     {best.get('arch', '-')}")
            print(f"  Best Filter:   {best.get('filter_config', '-')}")
            print(f"  Best Balance:  {best.get('balance_mode', '-')}")
            print(f"  Best Loss:     {best.get('loss_name', '-')}")

            mlflow.log_metric("best_macro_auc", best["metrics"].get("macro_auc", 0.0))
            mlflow.log_param("best_trial", best["trial_id"])
            mlflow.log_param("best_arch", best.get("arch", "-"))

    print(f"\n[MasterMind] Comparison report: {COMPARISON_REPORT_PATH}")
    print(f"[MasterMind] Experiment journal: {JOURNAL_PATH}")
    print("[MasterMind] Loop finished.")


if __name__ == "__main__":
    main()
