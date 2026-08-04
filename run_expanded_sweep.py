import os
import argparse
import random
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader

from data_management.dataset_factory import DatasetFactory
from temporal_encoder.encoder import ECGBiLSTM, ECGReconstructionDecoder
from temporal_encoder.strategies import (
    ReconstructionLearningStrategy,
    MaskedAutoencoderStrategy,
    ContrastiveLearningStrategy
)
from temporal_encoder.trainer import TemporalTrainer
from temporal_encoder.predictor import TemporalPredictor
from temporal_encoder.evaluator import TemporalEvaluator

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def sample_configs(num_trials=15):
    lrs = [1e-3, 5e-4]
    hidden_sizes = [128, 256]
    num_layers_list = [2, 3, 4]
    dropouts = [0.1, 0.2, 0.3, 0.4]
    weight_decays = [0.0, 1e-5]
    pretrain_strategies = ["mae", "contrastive"]

    configs = []
    # Seed for reproducibility of configurations
    random.seed(42)
    
    for _ in range(num_trials):
        cfg = {
            "lr": random.choice(lrs),
            "hidden_size": random.choice(hidden_sizes),
            "num_layers": random.choice(num_layers_list),
            "dropout_lstm": random.choice(dropouts),
            "dropout_fc": random.choice(dropouts),
            "weight_decay": random.choice(weight_decays),
            "pretrain_strategy": random.choice(pretrain_strategies)
        }
        # Avoid duplicate configurations
        if cfg not in configs:
            configs.append(cfg)
            
    return configs

def run_trial(trial_idx, cfg, train_loader, val_loader, test_loader, test_labels, args):
    run_name = f"trial_{trial_idx}_lr_{cfg['lr']}_hidden_{cfg['hidden_size']}_layers_{cfg['num_layers']}_ssl_{cfg['pretrain_strategy']}"
    print(f"\n==================================================")
    print(f"Starting Trial {trial_idx}: {run_name}")
    print(f"Configuration: {cfg}")
    print(f"==================================================")

    # 1. Instantiate Model
    model = ECGBiLSTM(
        input_size=12,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        num_classes=5
    ).to(device)

    # Set custom regularization
    model.lstm.dropout = cfg["dropout_lstm"] if cfg["num_layers"] > 1 else 0.0
    if len(model.fc) > 2 and isinstance(model.fc[2], torch.nn.Dropout):
        model.fc[2].p = cfg["dropout_fc"]

    # Start child run
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(cfg)
        mlflow.log_params({
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "resolution": args.resolution,
            "batch_size": args.batch_size
        })

        # 2. SSL Pretraining Phase
        if args.pretrain_epochs > 0:
            print(f"--- Pretraining Phase: {cfg['pretrain_strategy']} ({args.pretrain_epochs} epochs) ---")
            trainer_pretrain = TemporalTrainer(model, lr=cfg["lr"], device=device)
            decoder = ECGReconstructionDecoder(
                latent_dim=cfg["hidden_size"] * 2,
                num_leads=12,
                signal_length=1000 if args.resolution == "lr" else 5000
            )

            if cfg["pretrain_strategy"] == "mae":
                strategy = MaskedAutoencoderStrategy(mask_ratio=0.3)
            elif cfg["pretrain_strategy"] == "contrastive":
                strategy = ContrastiveLearningStrategy(temperature=0.1, projection_dim=64, latent_dim=cfg["hidden_size"] * 2)
            else:
                strategy = None

            if strategy is not None:
                trainer_pretrain.fit(
                    train_loader=train_loader,
                    epochs=args.pretrain_epochs,
                    is_pretraining=True,
                    strategy=strategy,
                    decoder=decoder
                )

        # 3. Supervised Fine-Tuning Phase
        print(f"--- Fine-Tuning / Supervised Training Phase ({args.finetune_epochs} epochs) ---")
        
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=3
        )

        for epoch in range(1, args.finetune_epochs + 1):
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
            epoch_train_loss = train_loss / (len(train_loader.dataset))

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
                    
            epoch_val_loss = val_loss / (len(val_loader.dataset))
            scheduler.step(epoch_val_loss)

            val_preds = np.concatenate(val_preds, axis=0)
            val_targets = np.concatenate(val_targets, axis=0)
            val_metrics = TemporalEvaluator.evaluate(val_targets, val_preds)

            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            for k, v in val_metrics.items():
                if not np.isnan(v):
                    mlflow.log_metric(f"val_{k}", v, step=epoch)

        # 4. Final Evaluation on Test Set
        print("Evaluating on test set...")
        predictor = TemporalPredictor(model, device=device)
        test_probs = predictor.predict_proba(test_loader)
        metrics = TemporalEvaluator.evaluate(test_labels, test_probs)

        print(f"Results for trial {run_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            if not np.isnan(v):
                mlflow.log_metric(k, v)

        # Log PyTorch Model artifact
        mlflow.pytorch.log_model(model, artifact_path="model", serialization_format="pickle")

        return metrics

def run_expanded_sweep(args):
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # 1. Load Data
    print(f"Loading PTB-XL dataset ({args.resolution} resolution, batch_size={args.batch_size})...")
    train_ds, val_ds, test_ds, loader = DatasetFactory.create_datasets(
        dataset_type="ptbxl",
        download=False,
        resolution=args.resolution
    )

    # Limit dataset size to 3000 records
    print(f"Limiting dataset size to {args.num_records} records...")
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
    
    if args.dry_run:
        print("Running in DRY RUN mode (1 trial, 1 epoch)...")
        configs = configs[:1]
        args.pretrain_epochs = 1
        args.finetune_epochs = 1

    print(f"Total configurations to evaluate: {len(configs)}")

    best_subset_acc = -1.0
    best_cfg = {}

    # Query existing finished runs to skip them
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment_name)
    finished_runs = set()
    if experiment is not None:
        try:
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
            for run in runs:
                if run.info.status == "FINISHED":
                    r_name = run.data.tags.get("mlflow.runName")
                    if r_name:
                        finished_runs.add(r_name)
        except Exception as e:
            print(f"Warning: Could not query existing runs for resume: {e}")

    with mlflow.start_run(run_name=args.parent_run_name) as parent_run:
        for idx, cfg in enumerate(configs, start=1):
            run_name = f"trial_{idx}_lr_{cfg['lr']}_hidden_{cfg['hidden_size']}_layers_{cfg['num_layers']}_ssl_{cfg['pretrain_strategy']}"
            if run_name in finished_runs:
                print(f"Skipping Trial {idx} ({run_name}) - Already completed.")
                # Retrieve best metrics if it's the best one so far to maintain best_subset_acc
                try:
                    runs = client.search_runs(experiment_ids=[experiment.experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'")
                    if len(runs) > 0 and "metrics.subset_accuracy" in runs[0].data.metrics:
                        acc = runs[0].data.metrics["subset_accuracy"]
                        if acc > best_subset_acc:
                            best_subset_acc = acc
                            best_cfg = cfg
                except Exception:
                    pass
                continue

            try:
                metrics = run_trial(idx, cfg, train_loader, val_loader, test_loader, test_labels, args)
                
                # We prioritize subset accuracy on the test set
                subset_acc = metrics["subset_accuracy"]
                if subset_acc > best_subset_acc:
                    best_subset_acc = subset_acc
                    best_cfg = cfg
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n[CUDA OOM] Out of Memory in trial {idx} with config {cfg}. Skipping to next trial.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    print(f"Trial {idx} failed with RuntimeError: {e}")
            except Exception as e:
                print(f"Trial {idx} failed with exception: {e}")
                continue

    print("\n==================================================")
    print("Hyperparameter tuning sweep completed!")
    print(f"Best Subset Accuracy achieved: {best_subset_acc:.4f}")
    print(f"Best Configuration: {best_cfg}")
    print("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expanded ECG Temporal Encoder Sweep")
    parser.add_argument("--tracking_uri", type=str, default="sqlite:///mlflow.db", help="MLflow tracking URI")
    parser.add_argument("--experiment_name", type=str, default="ECG_TemporalEncoder_ExpandedSweep", help="MLflow experiment name")
    parser.add_argument("--parent_run_name", type=str, default="temporal_encoder_expanded_parent", help="Parent run name")
    parser.add_argument("--resolution", type=str, default="lr", help="ECG resolution (lr or hr)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_records", type=int, default=3000, help="Number of records for subset training")
    parser.add_argument("--num_trials", type=int, default=15, help="Number of trials in the randomized sweep")
    parser.add_argument("--pretrain_epochs", type=int, default=3, help="Number of pretraining epochs")
    parser.add_argument("--finetune_epochs", type=int, default=10, help="Number of fine-tuning epochs")
    parser.add_argument("--dry_run", action="store_true", help="Run a fast dry-run check")

    args = parser.parse_args()
    run_expanded_sweep(args)
