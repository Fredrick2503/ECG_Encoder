import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import argparse
import random
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader

from data_management.dataset_factory import DatasetFactory
from temporal_encoder.encoder_upgrades import ECGTransformer
from temporal_encoder.predictor import TemporalPredictor
from temporal_encoder.evaluator import TemporalEvaluator

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def sample_configs(num_trials=12):
    d_models = [64, 128, 256]
    nheads = [4, 8]
    num_layers_list = [2, 3, 4]
    dim_feedforwards = [256, 512]
    dropouts = [0.1, 0.2, 0.3, 0.4]
    lrs = [1e-3, 5e-4]
    weight_decays = [0.0, 1e-5]

    configs = []
    random.seed(42)
    
    for _ in range(num_trials * 2):  # Sample extra to filter down
        cfg = {
            "d_model": random.choice(d_models),
            "nhead": random.choice(nheads),
            "num_layers": random.choice(num_layers_list),
            "dim_feedforward": random.choice(dim_feedforwards),
            "dropout": random.choice(dropouts),
            "lr": random.choice(lrs),
            "weight_decay": random.choice(weight_decays)
        }
        # Enforce d_model % nhead == 0 (always True for our values, but good safeguard)
        if cfg["d_model"] % cfg["nhead"] == 0:
            if cfg not in configs:
                configs.append(cfg)
                
    return configs[:num_trials]

def run_trial(trial_idx, cfg, train_loader, val_loader, test_loader, test_labels, args):
    run_name = f"trial_{trial_idx}_dmodel_{cfg['d_model']}_nhead_{cfg['nhead']}_layers_{cfg['num_layers']}_lr_{cfg['lr']}"
    print(f"\n==================================================")
    print(f"Starting Trial {trial_idx}: {run_name}")
    print(f"Configuration: {cfg}")
    print(f"==================================================")

    model = ECGTransformer(
        input_size=12,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        num_classes=5
    ).to(device)

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(cfg)
        mlflow.log_params({
            "epochs": args.epochs,
            "resolution": args.resolution,
            "batch_size": args.batch_size
        })

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=3
        )

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

            # Validate Epoch
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
            
            print(f"  Epoch {epoch}/{args.epochs} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f} - Val Subset Acc: {val_metrics['subset_accuracy']:.4f}")

        # Final Evaluation on Test Set
        print("Evaluating on test set...")
        predictor = TemporalPredictor(model, device=device)
        test_probs = predictor.predict_proba(test_loader)
        metrics = TemporalEvaluator.evaluate(test_labels, test_probs)

        print(f"Results for trial {run_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            if not np.isnan(v):
                mlflow.log_metric(f"test_{k}", v)

        # Log model
        mlflow.pytorch.log_model(model, artifact_path="model", serialization_format="pickle")

        return metrics

def main():
    parser = argparse.ArgumentParser(description="ECG Transformer Hyperparameter Sweep")
    parser.add_argument("--tracking_uri", type=str, default="file:///c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/mlruns", help="MLflow tracking URI")
    parser.add_argument("--experiment_name", type=str, default="ECG_Transformer_Tuning", help="MLflow experiment name")
    parser.add_argument("--resolution", type=str, default="lr", help="ECG resolution (lr or hr)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_records", type=int, default=3000, help="Number of records to use")
    parser.add_argument("--num_trials", type=int, default=12, help="Number of random configurations to test")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs per trial")
    parser.add_argument("--dry_run", action="store_true", help="Run a quick 1-epoch, 1-trial test")

    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN active. Testing 1 trial, 1 epoch on 200 records.")
        args.num_trials = 1
        args.epochs = 1
        args.num_records = 200

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # 1. Load Data
    print(f"Loading PTB-XL dataset ({args.resolution} resolution)...")
    train_ds, val_ds, test_ds, loader = DatasetFactory.create_datasets(
        dataset_type="ptbxl",
        download=False,
        resolution=args.resolution
    )

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

    # Sample configurations
    configs = sample_configs(num_trials=args.num_trials)
    print(f"Total configurations to evaluate: {len(configs)}")

    best_subset_acc = -1.0
    best_cfg = {}

    with mlflow.start_run(run_name="transformer_sweep_parent") as parent_run:
        for idx, cfg in enumerate(configs, start=1):
            try:
                metrics = run_trial(idx, cfg, train_loader, val_loader, test_loader, test_labels, args)
                subset_acc = metrics["subset_accuracy"]
                if subset_acc > best_subset_acc:
                    best_subset_acc = subset_acc
                    best_cfg = cfg
            except Exception as e:
                print(f"Trial {idx} failed with error: {e}")
                continue

    print("\n==================================================")
    print("Transformer hyperparameter sweep completed!")
    print(f"Best Test Subset Accuracy: {best_subset_acc:.4f}")
    print(f"Best Configuration: {best_cfg}")
    print("==================================================")

if __name__ == "__main__":
    main()
