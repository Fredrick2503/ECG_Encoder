import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import mlflow

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from classification.classifier import ZFusedDataset, LinearProbeClassifier, MLPClassifier
from classification.losses import BCEWithLogitsLoss, ClassBalancedLoss, AsymmetricLoss
from classification.metrics import calculate_metrics

# Settings
BATCH_SIZE = 64
EPOCHS = 30
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def optimize_thresholds(labels, probs):
    """
    Optimizes per-class decision thresholds on the validation set to maximize F1-score.
    """
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

def train_one_model(model, train_loader, val_loader, loss_fn, lr=1e-3, epochs=EPOCHS):
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
            
        # Validation evaluation
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
        
        # Optimize threshold and compute metrics
        thrs = optimize_thresholds(val_labels, val_probs)
        metrics = calculate_metrics(val_labels, val_probs, thrs)
        
        if metrics["macro_f1"] > best_val_f1:
            best_val_f1 = metrics["macro_f1"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    # Load best state
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_2k", action="store_true", help="Use 2K subset instead of full dataset")
    args = parser.parse_args()
    
    mlflow.set_tracking_uri("sqlite:///mlflow_benchmark.db")
    mlflow.set_experiment("ECG_Unified_Classification")
    
    # Load dataset
    data_file = project_root / "data" / ("Z_fused_2k.npz" if args.use_2k else "Z_fused_full.npz")
    print(f"Loading representation dataset from {data_file}...")
    
    data = np.load(data_file)
    
    # Extract representation segments
    # Dim 1056 is structured as: T (0-512), M (512-1024), B (1024-1056)
    train_z = data["train_z_fused"]
    val_z = data["val_z_fused"]
    test_z = data["test_z_fused"]
    
    train_y = data["train_labels"]
    val_y = data["val_labels"]
    test_y = data["test_labels"]
    
    print(f"Train size: {train_z.shape[0]}, Val size: {val_z.shape[0]}, Test size: {test_z.shape[0]}")
    
    # Compute samples per class for Class-Balanced loss
    class_counts = train_y.sum(axis=0)
    print(f"Class counts in training: {class_counts}")
    
    train_ds = ZFusedDataset(train_z, train_y)
    val_ds = ZFusedDataset(val_z, val_y)
    test_ds = ZFusedDataset(test_z, test_y)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    results = []
    
    # ─── C0 BASELINE: Linear Probe with BCE Loss ────────────────────────────────
    print("\n--- Training C0 Linear Probe Baseline ---")
    with mlflow.start_run(run_name="C0_Linear_Probe"):
        model_c0 = LinearProbeClassifier(input_dim=1056, num_classes=5)
        loss_fn = BCEWithLogitsLoss()
        
        model_c0 = train_one_model(model_c0, train_loader, val_loader, loss_fn)
        
        # Optimize threshold on Validation
        val_probs, val_labels = evaluate_model(model_c0, val_loader)
        thrs = optimize_thresholds(val_labels, val_probs)
        
        # Evaluate on Test
        test_probs, test_labels = evaluate_model(model_c0, test_loader)
        metrics = calculate_metrics(test_labels, test_probs, thrs)
        
        print(f"C0 Linear Probe Test Results:")
        print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
        print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
        print(f"  Subset Acc: {metrics['subset_acc']:.4f}")
        print(f"  Macro ECE: {metrics['macro_ece']:.4f}")
        
        # Log to MLflow
        mlflow.log_param("model_type", "C0_Linear_Probe")
        mlflow.log_param("dataset", "2k" if args.use_2k else "full")
        for k, v in metrics.items():
            if isinstance(v, list):
                for i, score in enumerate(v):
                    mlflow.log_metric(f"test_class_{i}_{k}", score)
            else:
                mlflow.log_metric(f"test_{k}", v)
                
        metrics["model"] = "C0_Linear_Probe"
        metrics["loss"] = "BCE"
        results.append(metrics)
        
        # Save model checkpoint
        os.makedirs(project_root / "models" / "exported", exist_ok=True)
        torch.save(model_c0.state_dict(), project_root / "models/exported/c0_linear_probe.pt")

    # ─── C1 MLP LOSS BENCHMARK: BCE vs CB-BCE vs ASL ──────────────────────────
    losses = {
        "BCE": BCEWithLogitsLoss(),
        "CB-BCE": ClassBalancedLoss(samples_per_class=class_counts, num_classes=5),
        "ASL": AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    }
    
    best_c1_macro_f1 = -1.0
    best_c1_model = None
    best_c1_loss_name = ""
    best_c1_metrics = None
    best_c1_thrs = None
    
    for loss_name, loss_fn in losses.items():
        print(f"\n--- Training C1 MLP with {loss_name} Loss ---")
        with mlflow.start_run(run_name=f"C1_MLP_{loss_name}"):
            model_c1 = MLPClassifier(input_dim=1056, hidden_dim=256, num_classes=5)
            model_c1 = train_one_model(model_c1, train_loader, val_loader, loss_fn)
            
            # Evaluate on Val
            val_probs, val_labels = evaluate_model(model_c1, val_loader)
            thrs = optimize_thresholds(val_labels, val_probs)
            
            # Evaluate on Test
            test_probs, test_labels = evaluate_model(model_c1, test_loader)
            metrics = calculate_metrics(test_labels, test_probs, thrs)
            
            print(f"C1 MLP {loss_name} Test Results:")
            print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
            print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
            print(f"  Subset Acc: {metrics['subset_acc']:.4f}")
            
            mlflow.log_param("model_type", "C1_MLP")
            mlflow.log_param("loss_type", loss_name)
            for k, v in metrics.items():
                if not isinstance(v, list):
                    mlflow.log_metric(f"test_{k}", v)
                    
            metrics["model"] = f"C1_MLP"
            metrics["loss"] = loss_name
            results.append(metrics)
            
            if metrics["macro_f1"] > best_c1_macro_f1:
                best_c1_macro_f1 = metrics["macro_f1"]
                best_c1_model = model_c1
                best_c1_loss_name = loss_name
                best_c1_metrics = metrics
                best_c1_thrs = thrs
                
    # Save best C1 model checkpoint
    print(f"\nBest C1 MLP Loss Configuration: {best_c1_loss_name} with Macro F1: {best_c1_macro_f1:.4f}")
    torch.save(best_c1_model.state_dict(), project_root / "models/exported/c1_mlp_best.pt")

    # ─── REPRESENTATION ABLATION BENCHMARK ──────────────────────────────────────
    # Dim boundaries: T (0-512), M (512-1024), B (1024-1056)
    ablation_configs = {
        "Temporal_Only (T)": (0, 512),
        "Morphology_Only (M)": (512, 1024),
        "Biomarker_Only (B)": (1024, 1056),
        "Pairwise_T_M": (0, 1024),
        "Pairwise_T_B": (0, 512, 1024, 1056),
        "Pairwise_M_B": (512, 1056),
        "Full_Fused (T_M_B)": (0, 1056)
    }
    
    print("\n--- Starting Representation Ablation Study ---")
    
    ablation_results = []
    
    for name, indices in ablation_configs.items():
        print(f"\nRunning Ablation: {name}")
        
        # Build index masks
        if len(indices) == 2:
            start, end = indices
            train_z_sub = train_z[:, start:end]
            val_z_sub = val_z[:, start:end]
            test_z_sub = test_z[:, start:end]
        else:
            # Pairwise T_B: concatenate T (0:512) and B (1024:1056)
            s1, e1, s2, e2 = indices
            train_z_sub = np.concatenate([train_z[:, s1:e1], train_z[:, s2:e2]], axis=1)
            val_z_sub = np.concatenate([val_z[:, s1:e1], val_z[:, s2:e2]], axis=1)
            test_z_sub = np.concatenate([test_z[:, s1:e1], test_z[:, s2:e2]], axis=1)
            
        sub_dim = train_z_sub.shape[1]
        
        train_ds_sub = ZFusedDataset(train_z_sub, train_y)
        val_ds_sub = ZFusedDataset(val_z_sub, val_y)
        test_ds_sub = ZFusedDataset(test_z_sub, test_y)
        
        train_loader_sub = DataLoader(train_ds_sub, batch_size=BATCH_SIZE, shuffle=True)
        val_loader_sub = DataLoader(val_ds_sub, batch_size=BATCH_SIZE, shuffle=False)
        test_loader_sub = DataLoader(test_ds_sub, batch_size=BATCH_SIZE, shuffle=False)
        
        with mlflow.start_run(run_name=f"Ablation_{name.replace(' ', '_')}"):
            # Train a linear probe on the ablation representation slice
            model_sub = LinearProbeClassifier(input_dim=sub_dim, num_classes=5)
            loss_fn = BCEWithLogitsLoss()
            model_sub = train_one_model(model_sub, train_loader_sub, val_loader_sub, loss_fn)
            
            # Evaluate
            val_probs, val_labels = evaluate_model(model_sub, val_loader_sub)
            thrs = optimize_thresholds(val_labels, val_probs)
            
            test_probs, test_labels = evaluate_model(model_sub, test_loader_sub)
            metrics = calculate_metrics(test_labels, test_probs, thrs)
            
            print(f"  {name} Test Results: Macro AUC: {metrics['macro_auc']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | Subset Acc: {metrics['subset_acc']:.4f}")
            
            mlflow.log_param("ablation_name", name)
            mlflow.log_param("input_dim", sub_dim)
            for k, v in metrics.items():
                if not isinstance(v, list):
                    mlflow.log_metric(f"test_{k}", v)
                    
            metrics["ablation"] = name
            ablation_results.append(metrics)

    # ─── SAVE REPORT ──────────────────────────────────────────────────────────
    print("\nSaving comparative reports...")
    
    # 1. Main model comparison table
    df_models = pd.DataFrame(results)
    df_models_clean = df_models[["model", "loss", "macro_auc", "macro_f1", "subset_acc", "macro_ece"]]
    
    # 2. Ablation comparison table
    df_ablation = pd.DataFrame(ablation_results)
    df_ablation_clean = df_ablation[["ablation", "macro_auc", "macro_f1", "subset_acc", "macro_ece"]]
    
    os.makedirs(project_root / "outputs/reports", exist_ok=True)
    report_path = project_root / "outputs/reports/classification_engine_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 4: Unified Classification Engine Benchmark Report\n\n")
        f.write(f"**Dataset Used**: {'PTB-XL 2K Subset' if args.use_2k else 'Full PTB-XL Dataset (21.8k records)'}\n\n")
        
        f.write("## 1. Classifier Model Comparison\n\n")
        f.write(df_models_clean.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 2. Representation Ablation Study (Linear Probe)\n\n")
        f.write(df_ablation_clean.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 3. Best Model Per-Class Detailed Performance\n\n")
        f.write(f"**Model**: C1 MLP Classifier | **Loss**: {best_c1_loss_name}\n\n")
        
        classes = ["NORM", "MI", "STTC", "CD", "HYP"]
        class_rows = []
        for i, c_name in enumerate(classes):
            class_rows.append({
                "Class": c_name,
                "F1-Score": best_c1_metrics["per_class_f1"][i],
                "ROC-AUC": best_c1_metrics["per_class_auc"][i],
                "ECE": best_c1_metrics["per_class_ece"][i],
                "Threshold": best_c1_thrs[i]
            })
        df_classes = pd.DataFrame(class_rows)
        f.write(df_classes.to_markdown(index=False))
        
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
