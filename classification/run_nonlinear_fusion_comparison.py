import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from classification.classifier import ZFusedDataset, MLPClassifier
from classification.losses import BCEWithLogitsLoss
from classification.metrics import calculate_metrics, compute_binary_ece
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, brier_score_loss

BATCH_SIZE = 64
EPOCHS = 30
SEEDS = [42, 100, 2026, 777, 999]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

def optimize_thresholds(labels, probs):
    num_classes = labels.shape[1]
    thresholds = np.full(num_classes, 0.5)
    for c in range(num_classes):
        best_f1, best_t = -1.0, 0.5
        for t in np.linspace(0.01, 0.99, 99):
            preds = (probs[:, c] >= t).astype(int)
            tp = np.sum((preds == 1) & (labels[:, c] == 1))
            fp = np.sum((preds == 1) & (labels[:, c] == 0))
            fn = np.sum((preds == 0) & (labels[:, c] == 1))
            f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[c] = best_t
    return thresholds

def train_model(model, train_loader, val_loader, loss_fn, lr=1e-3, epochs=EPOCHS):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    
    best_val_f1 = -1.0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            bz, by = batch["z"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(bz)
            loss = loss_fn(logits, by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                bz, by = batch["z"].to(device), batch["label"].to(device)
                probs = torch.sigmoid(model(bz))
                val_probs.append(probs.cpu().numpy())
                val_labels.append(by.cpu().numpy())
                
        val_probs = np.concatenate(val_probs, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        
        # Optimize validation threshold and calculate Macro F1
        thrs = optimize_thresholds(val_labels, val_probs)
        metrics = calculate_metrics(val_labels, val_probs, thrs)
        val_f1 = metrics["macro_f1"]
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    return model

def evaluate_model(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            bz, by = batch["z"].to(device), batch["label"].to(device)
            probs = torch.sigmoid(model(bz))
            all_probs.append(probs.cpu().numpy())
            all_labels.append(by.cpu().numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)

def run_experiment(train_z, val_z, test_z, train_y, val_y, test_y, input_dim, save_model_path=None, save_thrs_path=None):
    seed_runs = []
    loss_fn = BCEWithLogitsLoss()
    best_val_f1 = -1.0
    best_model_state = None
    best_thrs = None
    
    for seed in SEEDS:
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            
        train_loader = DataLoader(ZFusedDataset(train_z, train_y), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(ZFusedDataset(val_z, val_y), batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(ZFusedDataset(test_z, test_y), batch_size=BATCH_SIZE, shuffle=False)
        
        model = MLPClassifier(input_dim=input_dim, hidden_dim=256, num_classes=5)
        model = train_model(model, train_loader, val_loader, loss_fn)
        
        val_probs, val_labels = evaluate_model(model, val_loader)
        thrs = optimize_thresholds(val_labels, val_probs)
        
        # Track best model based on validation F1
        val_metrics = calculate_metrics(val_labels, val_probs, thrs)
        val_f1 = val_metrics["macro_f1"]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_thrs = thrs
            
        test_probs, test_labels = evaluate_model(model, test_loader)
        
        # Calculate metrics
        metrics = calculate_metrics(test_labels, test_probs, thrs)
        
        # Calculate Micro F1
        preds = (test_probs >= thrs).astype(int)
        micro_f1 = f1_score(test_labels, preds, average='micro')
        
        # Calculate mean Brier Score
        briers = [brier_score_loss(test_labels[:, c], test_probs[:, c]) for c in range(5)]
        mean_brier = np.mean(briers)
        
        run_data = {
            "seed": seed,
            "macro_auc": metrics["macro_auc"],
            "macro_f1": metrics["macro_f1"],
            "micro_f1": micro_f1,
            "subset_acc": metrics["subset_acc"],
            "macro_ece": metrics["macro_ece"],
            "brier": mean_brier,
            "per_class_f1": metrics["per_class_f1"]
        }
        seed_runs.append(run_data)
        
    if save_model_path is not None and best_model_state is not None:
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        torch.save(best_model_state, save_model_path)
        np.save(save_thrs_path, best_thrs)
        print(f"Saved best model checkpoint to {save_model_path} and thresholds to {save_thrs_path}")
        
    return seed_runs

def summarize_runs(runs):
    df = pd.DataFrame(runs)
    summary = {}
    for col in ["macro_auc", "macro_f1", "micro_f1", "subset_acc", "macro_ece", "brier"]:
        summary[col] = (df[col].mean(), df[col].std())
        
    # Per class summary
    per_class_list = np.array([r["per_class_f1"] for r in runs]) # (5, 5)
    summary["per_class_f1"] = [
        (float(per_class_list[:, c].mean()), float(per_class_list[:, c].std()))
        for c in range(5)
    ]
    return summary

def main():
    print("Loading PTB-XL pre-extracted representations splits...")
    data_file = project_root / "data" / "Z_fused_2k.npz"
    data = np.load(data_file)
    
    train_z = data["train_z_fused"]
    val_z = data["val_z_fused"]
    test_z = data["test_z_fused"]
    
    train_y = data["train_labels"]
    val_y = data["val_labels"]
    test_y = data["test_labels"]
    
    # Model A: T+M (Temporal + Morphology, 1024-D)
    print("\n--- Training Model A: T+M (Temporal + Morphology) across 5 seeds ---")
    train_z_a = train_z[:, 0:1024]
    val_z_a = val_z[:, 0:1024]
    test_z_a = test_z[:, 0:1024]
    runs_a = run_experiment(
        train_z_a, val_z_a, test_z_a, train_y, val_y, test_y, input_dim=1024,
        save_model_path=str(project_root / "models/classification_mlp_tm.pt"),
        save_thrs_path=str(project_root / "models/classification_mlp_tm_thresholds.npy")
    )
    summary_a = summarize_runs(runs_a)
    
    # Model B: T+M+B (Temporal + Morphology + Biomarker, 1056-D)
    print("\n--- Training Model B: T+M+B (Temporal + Morphology + Biomarker) across 5 seeds ---")
    runs_b = run_experiment(
        train_z, val_z, test_z, train_y, val_y, test_y, input_dim=1056,
        save_model_path=str(project_root / "models/classification_mlp.pt"),
        save_thrs_path=str(project_root / "models/classification_mlp_thresholds.npy")
    )
    summary_b = summarize_runs(runs_b)
    
    # Perform t-tests for statistical significance
    df_a = pd.DataFrame(runs_a)
    df_b = pd.DataFrame(runs_b)
    
    f1_t, f1_p = ttest_ind(df_a["macro_f1"], df_b["macro_f1"])
    ece_t, ece_p = ttest_ind(df_a["macro_ece"], df_b["macro_ece"])
    
    print("\n" + "="*50)
    print("Controlled Comparison Summary (Mean ± SD)")
    print("="*50)
    print(f"Metric          | Model A (T+M)        | Model B (T+M+B)")
    print("-"*50)
    for col in ["macro_auc", "macro_f1", "micro_f1", "subset_acc", "macro_ece", "brier"]:
        mean_a, std_a = summary_a[col]
        mean_b, std_b = summary_b[col]
        print(f"{col:<15} | {mean_a:.4f} ± {std_a:.4f}   | {mean_b:.4f} ± {std_b:.4f}")
    print("-"*50)
    print(f"t-test Macro F1: t = {f1_t:.4f}, p-value = {f1_p:.4f}")
    print(f"t-test Macro ECE: t = {ece_t:.4f}, p-value = {ece_p:.4f}")
    
    # Per-class summary print
    print("\nPer-class F1 Scores:")
    for c_idx, c_name in enumerate(CLASSES):
        mean_a, std_a = summary_a["per_class_f1"][c_idx]
        mean_b, std_b = summary_b["per_class_f1"][c_idx]
        print(f"  {c_name:<5} | Model A: {mean_a:.4f} ± {std_a:.4f} | Model B: {mean_b:.4f} ± {std_b:.4f}")
        
    # Plotting comparison figure
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_list = ["macro_f1", "macro_auc", "micro_f1", "subset_acc", "macro_ece"]
    means_a = [summary_a[m][0] for m in metrics_list]
    stds_a = [summary_a[m][1] for m in metrics_list]
    means_b = [summary_b[m][0] for m in metrics_list]
    stds_b = [summary_b[m][1] for m in metrics_list]
    
    x = np.arange(len(metrics_list))
    width = 0.35
    
    ax.bar(x - width/2, means_a, width, yerr=stds_a, label='Model A (T+M)', color='#1e88e5', capsize=5)
    ax.bar(x + width/2, means_b, width, yerr=stds_b, label='Model B (T+M+B)', color='#ff0d57', capsize=5)
    
    ax.set_title('Nonlinear Fusion Comparison: Model A (T+M) vs Model B (T+M+B)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics_list])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    fig_dir = project_root / "outputs/figures"
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = fig_dir / "nonlinear_fusion_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save Report
    report_dir = project_root / "outputs/reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = report_dir / "nonlinear_fusion_comparison.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 4: Controlled Nonlinear Fusion Benchmark Comparison\n\n")
        f.write("This report details the controlled comparison between Model A (T+M) and Model B (T+M+B) evaluated across 5 random seeds.\n\n")
        
        f.write("## 1. Aggregate Metrics Comparison (Mean ± SD)\n\n")
        f.write("| Metric | Model A (T+M) | Model B (T+M+B) | p-value |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for col in ["macro_auc", "macro_f1", "micro_f1", "subset_acc", "macro_ece", "brier"]:
            mean_a, std_a = summary_a[col]
            mean_b, std_b = summary_b[col]
            p_val = f1_p if col == "macro_f1" else (ece_p if col == "macro_ece" else np.nan)
            p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "-"
            f.write(f"| **{col.upper()}** | `{mean_a:.4f} ± {std_a:.4f}` | `{mean_b:.4f} ± {std_b:.4f}` | `{p_str}` |\n")
        f.write("\n\n")
        
        f.write("## 2. Per-Class F1 Scores Comparison\n\n")
        f.write("| Class | Model A (T+M) | Model B (T+M+B) |\n")
        f.write("| :--- | :---: | :---: |\n")
        for c_idx, c_name in enumerate(CLASSES):
            mean_a, std_a = summary_a["per_class_f1"][c_idx]
            mean_b, std_b = summary_b["per_class_f1"][c_idx]
            f.write(f"| **{c_name}** | `{mean_a:.4f} ± {std_a:.4f}` | `{mean_b:.4f} ± {std_b:.4f}` |\n")
        f.write("\n\n")
        
        f.write("## 3. Statistical and Clinical Conclusion\n\n")
        if mean_b > mean_a:
            f.write("### Retain Model B (T+M+B) with Biomarkers\n")
            f.write(f"Model B (T+M+B) outperforms Model A (T+M) on Macro F1 (`{mean_b:.4f}` vs `{mean_a:.4f}`). ")
            f.write("This confirms that tabular biomarkers supply incremental, non-linear clinical attributes that the multi-layer classification engine successfully leverages.\n")
        else:
            f.write("### Adopt Model A (T+M) as Primary Fusion\n")
            f.write(f"Model A (T+M) achieves higher or comparable Macro F1 (`{mean_a:.4f}` vs `{mean_b:.4f}`). ")
            f.write("Biomarkers do not supply sufficient incremental information to justify the extra model parameters. We select Model A as the primary predictive fusion pipeline.\n")
            
    print(f"Saved report to {report_path}")

if __name__ == "__main__":
    main()
