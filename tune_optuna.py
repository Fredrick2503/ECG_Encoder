import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import optuna

from data_management.dataset_factory import DatasetFactory
from temporal_encoder.encoder_upgrades import ECGResNet1D, ECGTransformer
from temporal_encoder.evaluator import TemporalEvaluator
from utils.losses import FocalLoss, AsymmetricLoss

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * signals.size(0)
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for signals, labels in loader:
            signals, labels = signals.to(device), labels.to(device)
            logits = model(signals)
            loss = criterion(logits, labels)
            total_loss += loss.item() * signals.size(0)
            probs = torch.sigmoid(logits)
            val_preds.append(probs.cpu().numpy())
            val_targets.append(labels.cpu().numpy())
            
    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)
    metrics = TemporalEvaluator.evaluate(val_targets, val_preds)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, metrics

def objective(trial, args, train_ds, val_ds):
    # 1. Suggest Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    scheduler_name = trial.suggest_categorical("scheduler", ["plateau", "cosine"])
    loss_type = trial.suggest_categorical("loss_type", ["bce", "focal", "asl"])
    
    asl_gamma_neg = 4.0
    asl_gamma_pos = 1.0
    if loss_type == "asl":
        asl_gamma_neg = trial.suggest_float("asl_gamma_neg", 1.0, 5.0)
        asl_gamma_pos = trial.suggest_float("asl_gamma_pos", 0.0, 2.0)
        
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # 2. Setup Dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3. Instantiate Model
    if args.model_type == "resnet":
        model = ECGResNet1D(
            input_size=12,
            num_classes=5,
            layers=[2, 2, 2, 2],
            base_filters=64,
            dropout=dropout,
            use_se=True # default to true for optimized resnet
        ).to(device)
    else:
        model = ECGTransformer(
            input_size=12,
            d_model=128,
            nhead=8,
            num_layers=3,
            dim_feedforward=256,
            dropout=dropout,
            num_classes=5
        ).to(device)

    # 4. Loss Function
    if loss_type == "focal":
        criterion = FocalLoss()
    elif loss_type == "asl":
        criterion = AsymmetricLoss(gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    # 5. Optimizer
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 6. Scheduler
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3)

    # MLflow tracking nested run
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True) as run:
        mlflow.log_params(trial.params)
        mlflow.log_param("model_type", args.model_type)
        
        best_val_f1 = -1.0
        
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_metrics = validate(model, val_loader, criterion)
            
            # Step scheduler
            if scheduler_name == "cosine":
                scheduler.step()
            else:
                scheduler.step(val_loss)
                
            val_f1 = val_metrics["macro_f1"]
            
            # Log metrics per epoch
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_macro_f1", val_f1, step=epoch)
            mlflow.log_metric("val_subset_accuracy", val_metrics["subset_accuracy"], step=epoch)
            mlflow.log_metric("val_macro_auc", val_metrics["macro_auc"], step=epoch)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                
            # Report intermediate values to Optuna for pruning if needed
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
        return best_val_f1

def main():
    parser = argparse.ArgumentParser(description="ECG Optuna Hyperparameter Optimization")
    parser.add_argument("--tracking_uri", type=str, default="file:///c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/mlruns", help="MLflow tracking URI")
    parser.add_argument("--experiment_name", type=str, default="ECG_Optuna_Tuning", help="MLflow experiment name")
    parser.add_argument("--model_type", type=str, default="resnet", choices=["resnet", "transformer"], help="Model backbone type")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs per trial")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials")
    
    args = parser.parse_args()
    
    # Setup MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    
    print("\nLoading PTB-XL dataset...")
    train_ds, val_ds, _, _ = DatasetFactory.create_datasets(
        dataset_type="ptbxl", download=False, resolution="lr"
    )
    print(f"Loaded records: Train={len(train_ds)}, Val={len(val_ds)}")
    
    with mlflow.start_run(run_name="optuna_study") as parent_run:
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, args, train_ds, val_ds),
            n_trials=args.n_trials
        )
        
        print("\n==================================================")
        print("OPTUNA TUNING STUDY COMPLETED")
        print("==================================================")
        print(f"Best Trial F1: {study.best_value:.4f}")
        print("Best Parameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
            mlflow.log_param(f"best_{k}", v)
        mlflow.log_metric("best_val_macro_f1", study.best_value)
        print("==================================================")

if __name__ == "__main__":
    main()
