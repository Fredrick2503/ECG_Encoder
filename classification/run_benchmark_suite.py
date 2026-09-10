"""
Comprehensive Benchmark Suite Runner.
Executes training, evaluation, and comparative benchmarking for the 6 target architectures:
1. CNN-LSTM (MIT-BIH Arrhythmia - AAMI 5-Class)
2. ECGFormer (MIT-BIH Arrhythmia - Transformer)
3. CNN-Transformer (MIT-BIH Arrhythmia - Hybrid)
4. Hybrid BERT-CNN (MIT-BIH Arrhythmia - Gated BERT+CNN)
5. FoundationalECGNet (PTB-XL 12-lead multi-label cardiac diagnosis)
6. RR Interval AF Detector (MIT-BIH AF Database - RR Rhythm Classification)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score

# Ensure root is on path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_management.mitbih_loader import get_mitbih_dataloaders
from data_management.mitbih_af_loader import get_mitbih_af_dataloaders
from data_management.dataset_factory import DatasetFactory
from models.benchmarks import (
    CNNLSTM,
    ECGFormer,
    CNNTransformer,
    HybridBERTCNN,
    FoundationalECGNet,
    RRAFDetector
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_REPORT_PATH = ROOT_DIR / "outputs" / "reports" / "benchmark_suite_report.md"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def train_epoch(model: nn.Module, loader, criterion, optimizer, is_multilabel: bool = False):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        # If [B, L, C] where L > C, transpose to [B, C, L]
        if x.dim() == 3 and x.shape[1] > x.shape[2] and not isinstance(model, RRAFDetector):
            x = x.transpose(1, 2)
        x = x.to(DEVICE).float()
        y = y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        if is_multilabel:
            loss = criterion(out, y.float())
        else:
            loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader, is_multilabel: bool = False) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []

    for x, y in loader:
        if x.dim() == 3 and x.shape[1] > x.shape[2] and not isinstance(model, RRAFDetector):
            x = x.transpose(1, 2)
        x = x.to(DEVICE).float()
        out = model(x)
        if is_multilabel:
            probs = torch.sigmoid(out).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
        else:
            probs = torch.softmax(out, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

        all_probs.append(probs)
        all_preds.append(preds)
        all_targets.append(y.numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    if is_multilabel:
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        try:
            auc = float(roc_auc_score(y_true, y_prob, average="macro"))
        except Exception:
            auc = 0.5
        recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    else:
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        try:
            if y_prob.shape[1] == 2:
                auc = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            auc = 0.5
        recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "roc_auc": auc,
        "sensitivity": recall
    }


def run_single_experiment(
    name: str,
    dataset_name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    epochs: int = 5,
    lr: float = 1e-3,
    is_multilabel: bool = False
) -> Dict[str, Any]:
    print(f"\n=======================================================", flush=True)
    print(f"[*] Starting Benchmark Trial: {name} on {dataset_name}", flush=True)
    print(f"=======================================================", flush=True)
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    if is_multilabel:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_auc = -1.0
    best_metrics = {}
    checkpoint_path = MODELS_DIR / f"benchmark_{name.lower().replace(' ', '_').replace('-', '_')}_best.pt"

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, is_multilabel=is_multilabel)
        val_metrics = evaluate_model(model, val_loader, is_multilabel=is_multilabel)
        scheduler.step(val_metrics["roc_auc"])

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {loss:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | Val F1: {val_metrics['macro_f1']*100:.2f}% | Val AUC: {val_metrics['roc_auc']*100:.2f}%")

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            torch.save(model.state_dict(), checkpoint_path)

    # Load best checkpoint and evaluate on Test set
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    test_metrics = evaluate_model(model, test_loader, is_multilabel=is_multilabel)
    elapsed = time.time() - start_time

    print(f"[+] Finished {name} | Test Acc: {test_metrics['accuracy']*100:.2f}% | Test F1: {test_metrics['macro_f1']*100:.2f}% | Test AUC: {test_metrics['roc_auc']*100:.2f}% (Elapsed: {elapsed:.1f}s)", flush=True)

    return {
        "model_name": name,
        "dataset": dataset_name,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_sensitivity": test_metrics["sensitivity"],
        "training_time_s": elapsed,
        "checkpoint": str(checkpoint_path)
    }


def main():
    parser = argparse.ArgumentParser(description="Run 6 Benchmark Experiments")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs per benchmark")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    results = []

    # -------------------------------------------------------------
    # 1. MIT-BIH Loaders (for CNN-LSTM, ECGFormer, CNN-Transformer, Hybrid BERT-CNN)
    # -------------------------------------------------------------
    print("Preparing MIT-BIH DataLoaders...", flush=True)
    mit_train_loader, mit_val_loader, mit_test_loader = get_mitbih_dataloaders(batch_size=args.batch_size)

    # 1. CNN-LSTM (MIT-BIH)
    m1 = CNNLSTM(in_channels=1, num_classes=5)
    r1 = run_single_experiment("CNN-LSTM", "MIT-BIH", m1, mit_train_loader, mit_val_loader, mit_test_loader, epochs=args.epochs)
    results.append(r1)

    # 2. ECGFormer (MIT-BIH)
    m2 = ECGFormer(in_channels=1, num_classes=5, patch_size=14, d_model=128, nhead=8, num_layers=3)
    r2 = run_single_experiment("ECGFormer", "MIT-BIH", m2, mit_train_loader, mit_val_loader, mit_test_loader, epochs=args.epochs)
    results.append(r2)

    # 3. CNN-Transformer (MIT-BIH)
    m3 = CNNTransformer(in_channels=1, num_classes=5, stem_channels=(32, 64, 128), d_model=128, nhead=8, num_layers=3)
    r3 = run_single_experiment("CNN-Transformer", "MIT-BIH", m3, mit_train_loader, mit_val_loader, mit_test_loader, epochs=args.epochs)
    results.append(r3)

    # 4. Hybrid BERT-CNN (MIT-BIH)
    m4 = HybridBERTCNN(in_channels=1, num_classes=5, cnn_hidden=64, d_model=128, nhead=8, num_layers=3)
    r4 = run_single_experiment("Hybrid BERT-CNN", "MIT-BIH", m4, mit_train_loader, mit_val_loader, mit_test_loader, epochs=args.epochs)
    results.append(r4)

    # -------------------------------------------------------------
    # 5. FoundationalECGNet (PTB-XL 12-lead multi-label)
    # -------------------------------------------------------------
    print("\nPreparing PTB-XL Foundation DataLoaders...", flush=True)
    try:
        ptb_train_loader, ptb_val_loader, ptb_test_loader, _ = DatasetFactory.create_dataloaders(
            dataset_type="ptbxl",
            resolution="lr",
            batch_size=args.batch_size,
            download=False
        )

        m5 = FoundationalECGNet(in_channels=12, num_classes=5, base_channels=48, d_model=128, nhead=8, num_trans_layers=2)
        r5 = run_single_experiment("FoundationalECGNet", "PTB-XL + CinC", m5, ptb_train_loader, ptb_val_loader, ptb_test_loader, epochs=args.epochs, is_multilabel=True)
        results.append(r5)
    except Exception as e:
        print(f"Warning: Failed running FoundationalECGNet: {e}", flush=True)

    # -------------------------------------------------------------
    # 6. RR Interval AF Detection (MIT-BIH AF)
    # -------------------------------------------------------------
    print("\nPreparing MIT-BIH AF DataLoaders...", flush=True)
    af_train_loader, af_val_loader, af_test_loader = get_mitbih_af_dataloaders(batch_size=args.batch_size, seq_len=50)
    m6 = RRAFDetector(in_dim=2, hidden_dim=64, num_layers=2, num_classes=2)
    r6 = run_single_experiment("RR Interval AF Detection", "MIT-BIH AF", m6, af_train_loader, af_val_loader, af_test_loader, epochs=args.epochs)
    results.append(r6)

    # -------------------------------------------------------------
    # Generate Benchmark Report
    # -------------------------------------------------------------
    report_lines = [
        "# Benchmark Experiments Suite Report\n",
        "## Summary of Evaluated Architectures and Datasets\n",
        "| Architecture / Method | Target Dataset | Test Accuracy | Test Macro F1 | Test ROC-AUC | Test Sensitivity | Time (s) | Checkpoint |",
        "|---|---|---|---|---|---|---|---|"
    ]

    for r in results:
        report_lines.append(
            f"| **{r['model_name']}** | {r['dataset']} | {r['test_accuracy']*100:.2f}% | {r['test_macro_f1']*100:.2f}% | {r['test_roc_auc']*100:.2f}% | {r['test_sensitivity']*100:.2f}% | {r['training_time_s']:.1f}s | `{Path(r['checkpoint']).name}` |"
        )

    report_lines.extend([
        "\n## Analysis & Findings\n",
        "1. **MIT-BIH Arrhythmia Benchmarks:** Evaluated AAMI 5-class beat classification across CNN-LSTM, ECGFormer, CNN-Transformer, and Hybrid BERT-CNN under standard inter-patient evaluation protocols.",
        "2. **FoundationalECGNet:** Evaluated 12-lead multi-label cardiac diagnosis on PTB-XL using hierarchical SE-ResNet temporal backbone and bidirectional cross-lead attention.",
        "3. **RR Interval AF Detection:** Assessed RR interval sequence dynamics and statistical HRV features on MIT-BIH AF rhythm classification.",
        "\nAll best model checkpoints are preserved in `models/`."
    ])

    report_content = "\n".join(report_lines)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n[+] Benchmark Suite Complete! Report saved to {OUTPUT_REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
