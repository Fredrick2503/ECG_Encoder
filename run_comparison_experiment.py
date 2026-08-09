import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import argparse
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader

from data_management.dataset_factory import DatasetFactory
from temporal_encoder.encoder_upgrades import ECGTransformer, ECGMultiScaleCNN
from temporal_encoder.predictor import TemporalPredictor
from temporal_encoder.evaluator import TemporalEvaluator

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def train_and_eval_model(model_class, model_name, model_kwargs, train_loader, val_loader, test_loader, test_labels, args):
    print(f"\n==================================================")
    print(f"Training {model_name}...")
    print(f"Arguments: {model_kwargs}")
    print(f"==================================================")
    
    model = model_class(**model_kwargs).to(device)
    
    # Optimizer and loss
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=3
    )
    
    # Active MLflow child run
    with mlflow.start_run(run_name=f"compare_{model_name.lower()}", nested=True):
        mlflow.log_params(model_kwargs)
        mlflow.log_params({
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "resolution": args.resolution,
            "model_type": model_name
        })
        
        best_val_loss = float("inf")
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            for signals, labels in train_loader:
                signals, labels = signals.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(signals)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * signals.size(0)
            epoch_train_loss = train_loss / len(train_loader.dataset)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for signals, labels in val_loader:
                    signals, labels = signals.to(device), labels.to(device)
                    logits = model(signals)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * signals.size(0)
                    
                    probs = torch.sigmoid(logits)
                    val_preds.append(probs.cpu().numpy())
                    val_targets.append(labels.cpu().numpy())
            
            epoch_val_loss = val_loss / len(val_loader.dataset)
            scheduler.step(epoch_val_loss)
            
            val_preds = np.concatenate(val_preds, axis=0)
            val_targets = np.concatenate(val_targets, axis=0)
            val_metrics = TemporalEvaluator.evaluate(val_targets, val_preds)
            
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            for k, v in val_metrics.items():
                if not np.isnan(v):
                    mlflow.log_metric(f"val_{k}", v, step=epoch)
            
            print(f"Epoch {epoch}/{args.epochs} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f} - Val Subset Acc: {val_metrics['subset_accuracy']:.4f}")
        
        # Test Evaluation
        print(f"Evaluating {model_name} on test set...")
        predictor = TemporalPredictor(model, device=device)
        test_probs = predictor.predict_proba(test_loader)
        test_metrics = TemporalEvaluator.evaluate(test_labels, test_probs)
        
        print(f"Test Results for {model_name}:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
            if not np.isnan(v):
                mlflow.log_metric(f"test_{k}", v)
        
        # Log Model
        mlflow.pytorch.log_model(model, artifact_path="model", serialization_format="pickle")
        
        return test_metrics

def main():
    parser = argparse.ArgumentParser(description="Compare Transformer vs Multi-Scale CNN")
    parser.add_argument("--tracking_uri", type=str, default="file:///c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/mlruns", help="MLflow tracking URI")
    parser.add_argument("--experiment_name", type=str, default="ECG_Architecture_Comparison", help="MLflow experiment name")
    parser.add_argument("--resolution", type=str, default="lr", help="ECG resolution (lr or hr)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_records", type=int, default=3000, help="Number of records to use")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--dry_run", action="store_true", help="Run a quick 1-epoch test")
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN active. Setting epochs to 1, dataset to 200 records.")
        args.epochs = 1
        args.num_records = 200

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    
    # Load Data
    print(f"Loading PTB-XL dataset ({args.resolution} resolution)...")
    train_ds, val_ds, test_ds, loader = DatasetFactory.create_datasets(
        dataset_type="ptbxl",
        download=False,
        resolution=args.resolution
    )
    
    # Slice subset
    train_ds.record_ids = train_ds.record_ids[:args.num_records]
    val_ds.record_ids = val_ds.record_ids[:max(5, int(args.num_records * 0.15))]
    test_ds.record_ids = test_ds.record_ids[:max(5, int(args.num_records * 0.15))]
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    test_labels = []
    for _, lbls in test_loader:
        test_labels.append(lbls.numpy())
    test_labels = np.concatenate(test_labels, axis=0)
    
    transformer_kwargs = {
        "input_size": 12,
        "d_model": 128,
        "nhead": 8,
        "num_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.3,
        "num_classes": 5
    }
    
    multiscale_kwargs = {
        "input_size": 12,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "num_classes": 5
    }
    
    # Parent Run
    with mlflow.start_run(run_name="architecture_comparison_parent") as parent_run:
        # 1. Train and evaluate Transformer
        transformer_metrics = train_and_eval_model(
            ECGTransformer, "ECGTransformer", transformer_kwargs,
            train_loader, val_loader, test_loader, test_labels, args
        )
        
        # 2. Train and evaluate Multi-Scale CNN
        multiscale_metrics = train_and_eval_model(
            ECGMultiScaleCNN, "ECGMultiScaleCNN", multiscale_kwargs,
            train_loader, val_loader, test_loader, test_labels, args
        )
        
        # Compare and print result
        print("\n==================================================")
        print("Comparison Results (Test Set):")
        print("==================================================")
        print(f"Transformer - Subset Accuracy: {transformer_metrics['subset_accuracy']:.4f}, Macro F1: {transformer_metrics['macro_f1']:.4f}, Macro AUC: {transformer_metrics['macro_auc']:.4f}")
        print(f"Multi-Scale - Subset Accuracy: {multiscale_metrics['subset_accuracy']:.4f}, Macro F1: {multiscale_metrics['macro_f1']:.4f}, Macro AUC: {multiscale_metrics['macro_auc']:.4f}")
        print("==================================================")
        
        # Write comparison details to markdown
        report_path = "C:/Users/fredr/.gemini/antigravity-ide/brain/3a6e217f-a003-4ed4-a18a-fe92e498191f/walkthrough.md"
        with open(report_path, "w") as f:
            f.write(f"""# Architecture Comparison Results

This document compares the results of upgrading the ECG Temporal Encoder to a **Transformer-based Encoder** vs. a **Multi-Scale CNN + BiLSTM Encoder**.

## Evaluation Summary (Test Set)

| Metric | ECGTransformer | ECGMultiScaleCNN | Winner |
| --- | --- | --- | --- |
| **Subset Accuracy** | {transformer_metrics['subset_accuracy']:.4f} | {multiscale_metrics['subset_accuracy']:.4f} | {"Transformer" if transformer_metrics['subset_accuracy'] > multiscale_metrics['subset_accuracy'] else "Multi-Scale CNN"} |
| **Hamming Loss** | {transformer_metrics['hamming_loss']:.4f} | {multiscale_metrics['hamming_loss']:.4f} | {"Transformer" if transformer_metrics['hamming_loss'] < multiscale_metrics['hamming_loss'] else "Multi-Scale CNN"} |
| **Macro F1-Score** | {transformer_metrics['macro_f1']:.4f} | {multiscale_metrics['macro_f1']:.4f} | {"Transformer" if transformer_metrics['macro_f1'] > multiscale_metrics['macro_f1'] else "Multi-Scale CNN"} |
| **Macro ROC-AUC** | {transformer_metrics['macro_auc']:.4f} | {multiscale_metrics['macro_auc']:.4f} | {"Transformer" if transformer_metrics['macro_auc'] > multiscale_metrics['macro_auc'] else "Multi-Scale CNN"} |

## Analysis
* **ECGTransformer:** Utilizes multi-head self-attention on the downsampled temporal sequence, capturing global context across the entire 10-second ECG recording.
* **ECGMultiScaleCNN:** Utilizes multiple receptive fields (5, 15, 51 time steps) to extract features of different durations (morphological complexes like QRS vs long slow waves) before recurrent processing with BiLSTM.
""")

if __name__ == "__main__":
    main()
