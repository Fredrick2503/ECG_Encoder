import os
import json
import time
import logging
import joblib
import pandas as pd
import numpy as np
import torch
from sklearn.manifold import TSNE

from biomarker_encoder.preprocessing import BiomarkerPreprocessor
from biomarker_encoder.models import AttentionMLPAutoencoder, BetaVAE, FTTransformerAutoencoder
from biomarker_encoder.trainer import BiomarkerTrainer
from biomarker_encoder.evaluator import BiomarkerEvaluator
from biomarker_encoder.tuning import run_optuna_study

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("BiomarkerExperiments")

def main():
    logger.info("Initializing Biomarker Comparison Experiment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Path configuration
    base_dir = "biomarker_encoder"
    output_dir = os.path.join(base_dir, "outputs")
    models_dir = os.path.join(base_dir, "models_checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    csv_path = "data/processed/full_biomarker_features.csv"
    if not os.path.exists(csv_path):
        logger.warning(f"Full biomarker features CSV not found at {csv_path}. Falling back to subset features...")
        csv_path = "previous_version/biomarker_encoder/ecg_features.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing features CSV at {csv_path}")
        
    # 1. Load and preprocess data
    preprocessor = BiomarkerPreprocessor(random_state=42)
    X_scaled, y, df = preprocessor.load_and_preprocess(csv_path)
    
    # Group by patient_id if available to prevent patient-level leakage
    patient_ids = df["patient_id"].values if "patient_id" in df.columns else None
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.get_splits(
        X_scaled, y, patient_ids=patient_ids
    )
    
    train_loader, val_loader, test_loader = preprocessor.get_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32
    )
    
    # Save preprocessor artifacts
    joblib.dump(preprocessor.scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(preprocessor.imputer, os.path.join(models_dir, "imputer.pkl"))
    joblib.dump(preprocessor.feature_cols, os.path.join(models_dir, "feature_cols.pkl"))
    logger.info(f"Preprocessor artifacts saved to {models_dir}")
    
    input_dim = X_scaled.shape[1]
    logger.info(f"Data preprocessed. Input Dimension: {input_dim}")
    logger.info(f"Splits size - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    model_types = ["attention_mlp", "beta_vae", "ft_transformer"]
    best_params = {}
    
    # 2. Hyperparameter Tuning
    # To run reasonably fast while being effective, we do 5 trials per model
    logger.info("Starting Hyperparameter Tuning...")
    for model_type in model_types:
        params = run_optuna_study(
            model_type=model_type,
            input_dim=input_dim,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=5,
            epochs=10
        )
        best_params[model_type] = params
        
    # Save tuned hyperparameters
    with open(os.path.join(output_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(best_params, f, indent=4)
    logger.info("Hyperparameter tuning completed and saved.")
    
    # 3. Model Training & Evaluation
    comparison_results = []
    evaluator = BiomarkerEvaluator(device=device)
    
    # Track embeddings and outputs for t-SNE / plotting
    visualizations_data = {}
    
    for model_type in model_types:
        logger.info(f"====================================================")
        logger.info(f"Training and Evaluating: {model_type}")
        logger.info(f"====================================================")
        
        params = best_params[model_type]
        latent_dim = params.get("latent_dim", 32)
        lr = params.get("lr", 1e-3)
        weight_decay = params.get("weight_decay", 1e-4)
        
        # Instantiate model with optimal params
        if model_type == "attention_mlp":
            hidden_units = params.get("hidden_units", 128)
            num_heads = params.get("num_heads", 4)
            dropout = params.get("dropout", 0.3)
            # Ensure divisibility
            if hidden_units % num_heads != 0:
                hidden_units = (hidden_units // num_heads) * num_heads
                if hidden_units == 0:
                    hidden_units = num_heads
            model = AttentionMLPAutoencoder(
                input_dim=input_dim,
                latent_dim=latent_dim,
                dropout=dropout,
                num_heads=num_heads,
                hidden_units=hidden_units
            )
        elif model_type == "beta_vae":
            hidden_units = params.get("hidden_units", 128)
            beta = params.get("beta", 1.0)
            model = BetaVAE(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_units=hidden_units,
                beta=beta
            )
        elif model_type == "ft_transformer":
            d_model = params.get("d_model", 64)
            nhead = params.get("nhead", 4)
            num_layers = params.get("num_layers", 2)
            ffn_dim = params.get("ffn_dim", 128)
            dropout = params.get("dropout", 0.2)
            model = FTTransformerAutoencoder(
                input_dim=input_dim,
                latent_dim=latent_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                ffn_dim=ffn_dim,
                dropout=dropout
            )
            
        checkpoint_path = os.path.join(models_dir, f"{model_type}_best.pt")
        trainer = BiomarkerTrainer(
            model=model,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            patience=15,
            checkpoint_path=checkpoint_path,
            mixed_precision=True
        )
        
        # Train model
        train_start = time.time()
        _, train_losses, val_losses = trainer.fit(train_loader, val_loader, epochs=60)
        training_time = time.time() - train_start
        
        # Evaluate model
        metrics, test_embeddings, test_reconstructed, test_inputs = evaluator.evaluate_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            y_train=y_train,
            y_test=y_test
        )
        
        # Log training time and history
        metrics["model_type"] = model_type
        metrics["total_training_time"] = training_time
        metrics["final_train_loss"] = train_losses[-1]
        metrics["final_val_loss"] = val_losses[-1]
        
        comparison_results.append(metrics)
        
        # Calculate t-SNE coordinates for visualization
        logger.info(f"Computing t-SNE for {model_type}...")
        perplexity = min(30, max(1, len(test_embeddings) - 1))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_coords = tsne.fit_transform(test_embeddings)
        
        visualizations_data[model_type] = {
            "embeddings": test_embeddings.tolist(),
            "tsne_x": tsne_coords[:, 0].tolist(),
            "tsne_y": tsne_coords[:, 1].tolist(),
            "reconstructed": test_reconstructed.tolist(),
            "inputs": test_inputs.tolist()
        }
        
    # 4. Generate Reports
    results_df = pd.DataFrame(comparison_results)
    results_df.to_csv(os.path.join(output_dir, "model_comparison_metrics.csv"), index=False)
    
    # Save visualizations data for visualization notebook
    with open(os.path.join(output_dir, "visualization_data.json"), "w") as f:
        json.dump(visualizations_data, f)
        
    # Write Markdown Report
    best_model_idx = results_df["MSE"].idxmin()
    best_model_name = results_df.loc[best_model_idx, "model_type"]
    
    report_path = os.path.join(output_dir, "benchmarking_report.md")
    with open(report_path, "w") as f:
        f.write("# ECG Biomarker Encoder Benchmarking Report\n\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"We trained, tuned, and compared three latent representation learning models on {len(X_scaled)} ECG biomarker feature profiles. ")
        f.write("The models were updated to support joint feature reconstruction and direct diagnostic classification using imputed inputs + binary missingness masks.\n\n")
        f.write(f"Based on reconstruction error (MSE), **{best_model_name}** is the recommended model.\n\n")
        
        f.write("## Performance Metrics Comparison\n\n")
        f.write("| Model Type | Params | Reconstruction MSE | Reconstruction MAE | Latent Silhouette | Downstream F1 Score | Downstream ROC-AUC | Direct F1 Score | Direct ROC-AUC | Training Time (s) | Inference Time / Sample (s) |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for _, row in results_df.iterrows():
            f.write(
                f"| {row['model_type']} | {row['num_parameters']:,} | {row['MSE']:.6f} | {row['MAE']:.6f} | "
                f"{row['Silhouette_Score']:.4f} | {row['Downstream_F1_Score']:.4f} | {row['Downstream_ROC_AUC']:.4f} | "
                f"{row['Direct_F1_Score']:.4f} | {row['Direct_ROC_AUC']:.4f} | "
                f"{row['total_training_time']:.2f} | {row['inference_time_per_sample']:.6f} |\n"
            )
        f.write("\n")
        
        f.write("## Recommendation & Analysis\n\n")
        f.write(f"1. **Reconstruction Quality**: `{best_model_name}` achieved the lowest mean squared error on the test dataset. ")
        f.write("A lower MSE indicates the learned latent space preserves the details of the input biomarkers.\n")
        f.write("2. **Latent Space Clusterability**: The Silhouette Score evaluates how well the latent representation aligns with clinical labels. ")
        f.write("Models with positive Silhouette scores learn structured manifolds that reflect downstream pathology.\n")
        f.write("3. **Downstream Classification**: Training a simple classifier directly on the latent 32-dim representation verifies the downstream clinical utility of the representation.\n")
        f.write("4. **Direct Classification**: Evaluating the model's internal classification head shows how well the model jointly learns reconstruction and classification.\n")
        
    logger.info("====================================================")
    logger.info("Experiment Completed Successfully!")
    logger.info(f"Benchmarking report saved to {report_path}")
    logger.info(f"Recommended model: {best_model_name}")
    logger.info("====================================================")

if __name__ == "__main__":
    main()
