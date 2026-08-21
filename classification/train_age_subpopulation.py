import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
import pickle
import copy
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from data_management.dataset_factory import DatasetFactory
from classification.classifier import ZFusedDataset, MLPClassifier
from classification.losses import BCEWithLogitsLoss
from classification.metrics import calculate_metrics

BATCH_SIZE = 32
EPOCHS = 35
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

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

def train_pipeline_grid_search(input_dim, train_z, train_y, val_z, val_y, test_z, test_y, model_name, model_file, thr_file):
    print(f"\nGrid Search for {model_name} (input_dim={input_dim})...")
    train_loader = DataLoader(ZFusedDataset(train_z, train_y), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ZFusedDataset(val_z, val_y), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ZFusedDataset(test_z, test_y), batch_size=BATCH_SIZE, shuffle=False)
    
    # Expanded Grid Search Candidates
    lrs = [2e-3, 1e-3, 5e-4]
    dropouts = [0.2, 0.3, 0.4]
    hidden_dims = [64, 128, 256]
    weight_decays = [1e-3, 1e-4]
    
    best_overall_val_f1 = -1.0
    best_overall_state = None
    best_config = {}
    
    for lr in lrs:
        for do in dropouts:
            for hd in hidden_dims:
                for wd in weight_decays:
                    model = MLPClassifier(input_dim=input_dim, hidden_dim=hd, num_classes=5, dropout=do).to(device)
                    loss_fn = BCEWithLogitsLoss().to(device)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
                    
                    best_val_f1 = -1.0
                    best_state = None
                    
                    for epoch in range(EPOCHS):
                        model.train()
                        for batch in train_loader:
                            bz, by = batch["z"].to(device), batch["label"].to(device)
                            optimizer.zero_grad()
                            logits = model(bz)
                            loss = loss_fn(logits, by)
                            loss.backward()
                            optimizer.step()
                        scheduler.step()
                        
                        # Eval on Val
                        model.eval()
                        val_probs = []
                        with torch.no_grad():
                            for batch in val_loader:
                                bz = batch["z"].to(device)
                                probs = torch.sigmoid(model(bz))
                                val_probs.append(probs.cpu().numpy())
                        val_probs = np.concatenate(val_probs, axis=0)
                        
                        val_metrics = calculate_metrics(val_y, val_probs, np.full(5, 0.5))
                        if val_metrics["macro_f1"] > best_val_f1:
                            best_val_f1 = val_metrics["macro_f1"]
                            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    
                    if best_val_f1 > best_overall_val_f1:
                        best_overall_val_f1 = best_val_f1
                        best_overall_state = best_state
                        best_config = {"lr": lr, "dropout": do, "hidden_dim": hd, "weight_decay": wd}
                        
    print(f"  Best Config for {model_name}: {best_config} with Val F1={best_overall_val_f1:.4f}")
    
    # Instantiate best model
    model = MLPClassifier(
        input_dim=input_dim, 
        hidden_dim=best_config["hidden_dim"], 
        num_classes=5, 
        dropout=best_config["dropout"]
    ).to(device)
    model.load_state_dict({k: v.to(device) for k, v in best_overall_state.items()})
    model.eval()
    
    # Threshold optimization on train set
    train_loader_eval = DataLoader(ZFusedDataset(train_z, train_y), batch_size=BATCH_SIZE, shuffle=False)
    train_probs = []
    with torch.no_grad():
        for batch in train_loader_eval:
            bz = batch["z"].to(device)
            probs = torch.sigmoid(model(bz))
            train_probs.append(probs.cpu().numpy())
    train_probs = np.concatenate(train_probs, axis=0)
    thrs = optimize_thresholds(train_y, train_probs)
    
    # Test evaluation
    test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            bz = batch["z"].to(device)
            probs = torch.sigmoid(model(bz))
            test_probs.append(probs.cpu().numpy())
    test_probs = np.concatenate(test_probs, axis=0)
    
    metrics = calculate_metrics(test_y, test_probs, thrs)
    print(f"{model_name} Test Cohort Metrics:")
    print(f"  Macro F1: {metrics['macro_f1']:.4f} | Subset Accuracy: {metrics['subset_acc']:.4f} | ECE: {metrics['macro_ece']:.4f}")
    
    # Save
    model_dir = project_root / "models"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), model_dir / model_file)
    np.save(model_dir / thr_file, thrs)
    print(f"Saved {model_name} checkpoint and thresholds.")
    
    return metrics

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow_benchmark.db")
    mlflow.set_experiment("ECG_Age_18_30_Subpopulation")
    
    print("Loading metadata...")
    _, _, _, loader = DatasetFactory.create_datasets(dataset_type="ptbxl", download=False, resolution="lr")
    df = loader.metadata_df
    
    data_file = project_root / "data" / "Z_fused_2k.npz"
    data = np.load(data_file)
    
    train_z, train_y, train_rids = data["train_z_fused"], data["train_labels"], data["train_record_id"]
    val_z, val_y, val_rids = data["val_z_fused"], data["val_labels"], data["val_record_id"]
    test_z, test_y, test_rids = data["test_z_fused"], data["test_labels"], data["test_record_id"]
    
    def filter_age_cohort(z, y, rids):
        indices = []
        for idx, rid in enumerate(rids):
            rid_int = int(rid)
            if rid_int in df.index:
                age = df.loc[rid_int, "age"]
                if pd.notna(age) and 18 <= age <= 30:
                    indices.append(idx)
        return z[indices], y[indices], rids[indices]
        
    train_z_f, train_y_f, _ = filter_age_cohort(train_z, train_y, train_rids)
    val_z_f, val_y_f, _ = filter_age_cohort(val_z, val_y, val_rids)
    test_z_f, test_y_f, _ = filter_age_cohort(test_z, test_y, test_rids)
    
    with mlflow.start_run():
        # Train Model B (T+M+B) - 1056 dimensions
        metrics_b = train_pipeline_grid_search(
            1056, train_z_f, train_y_f, val_z_f, val_y_f, test_z_f, test_y_f,
            "Model B (T+M+B)", "classification_mlp_age_18_30.pt", "classification_mlp_age_18_30_thresholds.npy"
        )
        
        # Train Model A (T+M) - 1024 dimensions
        train_z_f_a = train_z_f[:, :1024]
        val_z_f_a = val_z_f[:, :1024]
        test_z_f_a = test_z_f[:, :1024]
        
        metrics_a = train_pipeline_grid_search(
            1024, train_z_f_a, train_y_f, val_z_f_a, val_y_f, test_z_f_a, test_y_f,
            "Model A (T+M)", "classification_mlp_age_18_30_tm.pt", "classification_mlp_age_18_30_tm_thresholds.npy"
        )
        
        # Write markdown report
        report_path = project_root / "outputs/reports/age_18_30_cohort_report.md"
        os.makedirs(report_path.parent, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Age 18-30 Cohort Fine-Tuning Performance Report\n\n")
            f.write("## 1. Split sizes\n")
            f.write(f"- Train size: {len(train_z_f)}\n")
            f.write(f"- Val size: {len(val_z_f)}\n")
            f.write(f"- Test size: {len(test_z_f)}\n\n")
            f.write("## 2. Test-set Performance Metrics\n")
            f.write("| Model | Macro F1 | Subset Accuracy | Macro ECE |\n")
            f.write("| --- | --- | --- | --- |\n")
            f.write(f"| Model A (T+M) | {metrics_a['macro_f1']:.4f} | {metrics_a['subset_acc']:.4f} | {metrics_a['macro_ece']:.4f} |\n")
            f.write(f"| Model B (T+M+B) | {metrics_b['macro_f1']:.4f} | {metrics_b['subset_acc']:.4f} | {metrics_b['macro_ece']:.4f} |\n")
            
        print(f"Saved cohort report to {report_path}")

if __name__ == "__main__":
    main()
