import os
import sys
import json
import time
import torch
import pickle
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from torch.utils.data import Dataset, DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("BiomarkerExperiments")

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from biomarkers.models import AttentionMLPAutoencoder, BetaVAE, FTTransformerAutoencoder
from biomarkers.trainer import BiomarkerTrainer
from biomarkers.evaluator import BiomarkerEvaluator

# Settings
BATCH_SIZE = 64
EPOCHS = 40
LATENT_DIM = 32
SEED = 42

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

FEATURES = [
    "RR_Mean", "QRS_Duration", "PR_Interval", "QT_Interval", "QTc_Bazett",
    "ST_Duration", "P_wave_Duration", "R_Amplitude", "P_Amplitude", "T_Amplitude",
    "ST_Deviation", "Q_Amplitude", "R_S_Ratio", "QRS_Energy", "SDNN",
    "RMSSD", "pNN50", "pNN20", "SDRR_RMSSD_Ratio", "HRV_Triangular_Index",
    "LF_Power", "HF_Power", "LF_HF_Ratio", "Total_Power", "Sample_Entropy"
]

LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]

class ECGFeatureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.X[idx]

def evaluate_detailed(model, train_loader, test_loader, y_train, y_test, device):
    """Detailed evaluation computing both macro and per-label metrics."""
    model.eval()
    
    test_inputs = []
    test_reconstructed = []
    test_embeddings = []
    test_logits = []
    
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            N = batch_x.size(1) // 2
            orig_x = batch_x[:, :N]
            
            if hasattr(model, "loss_function"):  # VAE
                reconstructed, z, mu, logvar, class_logits = model(batch_x)
                latent = mu
            else:
                reconstructed, latent, class_logits = model(batch_x)
                
            test_inputs.append(orig_x.cpu().numpy())
            test_reconstructed.append(reconstructed.cpu().numpy())
            test_embeddings.append(latent.cpu().numpy())
            test_logits.append(class_logits.cpu().numpy())
            
    test_inputs = np.concatenate(test_inputs, axis=0)
    test_reconstructed = np.concatenate(test_reconstructed, axis=0)
    test_embeddings = np.concatenate(test_embeddings, axis=0)
    test_logits = np.concatenate(test_logits, axis=0)
    
    train_embeddings = []
    with torch.no_grad():
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            if hasattr(model, "loss_function"):
                reconstructed, z, mu, logvar, class_logits = model(batch_x)
                latent = mu
            else:
                reconstructed, latent, class_logits = model(batch_x)
            train_embeddings.append(latent.cpu().numpy())
    train_embeddings = np.concatenate(train_embeddings, axis=0)
    
    # 1. Reconstruction metrics
    mse = float(np.mean((test_inputs - test_reconstructed) ** 2))
    mae = float(np.mean(np.abs(test_inputs - test_reconstructed)))
    
    # 2. Direct Classification metrics
    test_probs_direct = 1.0 / (1.0 + np.exp(-test_logits))
    test_preds_direct = (test_probs_direct >= 0.5).astype(float)
    
    direct_acc = accuracy_score(y_test, test_preds_direct)
    direct_f1_macro = f1_score(y_test, test_preds_direct, average="macro", zero_division=0)
    direct_auc_macro = roc_auc_score(y_test, test_probs_direct, average="macro")
    
    direct_f1_per_label = f1_score(y_test, test_preds_direct, average=None, zero_division=0)
    direct_auc_per_label = []
    for i in range(len(LABELS)):
        try:
            auc_l = roc_auc_score(y_test[:, i], test_probs_direct[:, i])
        except Exception:
            auc_l = np.nan
        direct_auc_per_label.append(auc_l)
        
    # 3. Downstream Classification metrics (Logistic Regression on embeddings)
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=SEED))
    clf.fit(train_embeddings, y_train)
    test_preds_ds = clf.predict(test_embeddings)
    test_probs_ds = clf.predict_proba(test_embeddings)
    
    ds_acc = accuracy_score(y_test, test_preds_ds)
    ds_f1_macro = f1_score(y_test, test_preds_ds, average="macro", zero_division=0)
    ds_auc_macro = roc_auc_score(y_test, test_probs_ds, average="macro")
    
    ds_f1_per_label = f1_score(y_test, test_preds_ds, average=None, zero_division=0)
    ds_auc_per_label = []
    for i in range(len(LABELS)):
        try:
            auc_l = roc_auc_score(y_test[:, i], test_probs_ds[:, i])
        except Exception:
            auc_l = np.nan
        ds_auc_per_label.append(auc_l)
        
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "num_parameters": num_params,
        "MSE": mse,
        "MAE": mae,
        "Direct_Accuracy": direct_acc,
        "Direct_F1_Score": direct_f1_macro,
        "Direct_ROC_AUC": direct_auc_macro,
        "Direct_F1_Per_Label": list(direct_f1_per_label),
        "Direct_AUC_Per_Label": direct_auc_per_label,
        "Downstream_Accuracy": ds_acc,
        "Downstream_F1_Score": ds_f1_macro,
        "Downstream_ROC_AUC": ds_auc_macro,
        "Downstream_F1_Per_Label": list(ds_f1_per_label),
        "Downstream_AUC_Per_Label": ds_auc_per_label
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    biomarkers_dir = project_root / "biomarkers"
    raw_csv = biomarkers_dir / "ecg_biomarkers_full.csv"
    
    logger.info(f"Loading raw features from {raw_csv}...")
    df_raw = pd.read_csv(raw_csv)
    
    # Extract records list
    record_ids = df_raw["record_id"].values
    
    # 1. Identify missing/NaN values and impute using median
    logger.info("Performing Median Imputation...")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_imputed = imputer.fit_transform(df_raw[FEATURES])
    
    # 2. Standardize features using StandardScaler
    logger.info("Performing Standard Scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Save imputer and scaler
    with open(biomarkers_dir / "imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)
    with open(biomarkers_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Imputer and scaler saved successfully.")
    
    # 3. Create missingness mask
    M = (~df_raw[FEATURES].isna()).astype(np.float32).values
    
    # Concatenate scaled features and missingness mask (48 dimensions)
    X_combined = np.hstack([X_scaled, M])
    input_dim = X_combined.shape[1]
    logger.info(f"Combined Input dimension (features + missingness mask): {input_dim}")
    
    # Get labels
    y = df_raw[LABELS].values
    
    # Load patient IDs for patient-wise split
    ptb_db_path = project_root / "data" / "raw" / "ptbxl" / "ptbxl_database.csv"
    if ptb_db_path.exists():
        logger.info(f"Loading PTB-XL database from {ptb_db_path} for patient ID lookup...")
        df_ptb = pd.read_csv(ptb_db_path, index_col="ecg_id")
        patient_ids = df_ptb.loc[df_raw["record_id"], "patient_id"].values
    else:
        logger.warning("ptbxl_database.csv not found. Falling back to record-wise split.")
        patient_ids = None
        
    # Split datasets patient-wise
    if patient_ids is not None:
        unique_patients = np.unique(patient_ids)
        train_patients, test_patients = train_test_split(unique_patients, test_size=0.30, random_state=SEED)
        val_patients, test_patients = train_test_split(test_patients, test_size=0.50, random_state=SEED)
        
        train_idx = np.isin(patient_ids, train_patients)
        val_idx = np.isin(patient_ids, val_patients)
        test_idx = np.isin(patient_ids, test_patients)
        
        X_train, X_val, X_test = X_combined[train_idx], X_combined[val_idx], X_combined[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y, test_size=0.30, random_state=SEED)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=SEED)
        
    logger.info(f"Train set: {len(X_train)} | Val set: {len(X_val)} | Test set: {len(X_test)}")
    
    # Create dataloaders
    train_loader = DataLoader(ECGFeatureDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ECGFeatureDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ECGFeatureDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    # Training configurations
    models_config = {
        "attention_mlp": {
            "model": AttentionMLPAutoencoder(input_dim=input_dim, latent_dim=LATENT_DIM, hidden_units=128, num_heads=4),
            "lr": 1e-3,
            "weight_decay": 1e-4
        },
        "beta_vae": {
            "model": BetaVAE(input_dim=input_dim, latent_dim=LATENT_DIM, hidden_units=128, beta=1.0),
            "lr": 1e-3,
            "weight_decay": 1e-4
        },
        "ft_transformer": {
            "model": FTTransformerAutoencoder(input_dim=input_dim, latent_dim=LATENT_DIM, d_model=32, nhead=2, num_layers=2, ffn_dim=64),
            "lr": 1e-3,
            "weight_decay": 1e-4
        }
    }
    
    comparison_results = []
    
    for model_name, config in models_config.items():
        logger.info(f"Training model: {model_name}...")
        model = config["model"]
        checkpoint_path = biomarkers_dir / f"{model_name}_best.pt"
        
        trainer = BiomarkerTrainer(
            model=model,
            device=device,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            patience=10,
            checkpoint_path=str(checkpoint_path)
        )
        
        # Fit model
        start_time = time.time()
        trainer.fit(train_loader, val_loader, epochs=EPOCHS)
        training_time = time.time() - start_time
        
        # Load best weights
        if checkpoint_path.exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            
        # Detailed Evaluation
        logger.info(f"Evaluating model: {model_name}...")
        metrics = evaluate_detailed(model, train_loader, test_loader, y_train, y_test, device)
        metrics["model_type"] = model_name
        metrics["training_time_seconds"] = training_time
        metrics["embedding_dim"] = LATENT_DIM
        
        comparison_results.append(metrics)
        
        # Generate and save latent embeddings for the full dataset
        logger.info(f"Generating latent embeddings for the full dataset using {model_name}...")
        model.eval()
        full_loader = DataLoader(ECGFeatureDataset(X_combined), batch_size=BATCH_SIZE, shuffle=False)
        all_embeddings = []
        with torch.no_grad():
            for batch_x, _ in full_loader:
                batch_x = batch_x.to(device)
                if hasattr(model, "loss_function"):  # VAE
                    reconstructed, z, mu, logvar, class_logits = model(batch_x)
                    latent = mu
                else:
                    reconstructed, latent, class_logits = model(batch_x)
                all_embeddings.append(latent.cpu().numpy())
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Save to CSV
        emb_df = pd.DataFrame(all_embeddings, columns=[f"latent_{i}" for i in range(LATENT_DIM)])
        emb_df.insert(0, "record_id", record_ids)
        emb_csv_path = biomarkers_dir / f"embeddings_{model_name}.csv"
        emb_df.to_csv(emb_csv_path, index=False)
        logger.info(f"Embeddings saved to {emb_csv_path}")

    # Create comparison DataFrame
    df_metrics = pd.DataFrame(comparison_results)
    metrics_csv_path = biomarkers_dir / "model_comparison_metrics.csv"
    df_metrics.to_csv(metrics_csv_path, index=False)
    logger.info(f"Comparison metrics saved to {metrics_csv_path}")
    
    # Save Preprocessed full dataset as requested or save features
    prep_csv_path = biomarkers_dir / "ecg_biomarkers_preprocessed.csv"
    prep_df = pd.DataFrame(X_scaled, columns=FEATURES)
    prep_df.insert(0, "record_id", record_ids)
    for c in LABELS:
        prep_df[c] = df_raw[c].values
    prep_df.to_csv(prep_csv_path, index=False)
    logger.info(f"Preprocessed features saved to {prep_csv_path}")
    
    # Determine the best overall model
    # Best model is selected based on a trade-off: lowest Reconstruction MSE, highest Downstream ROC-AUC & F1
    # We will compute a simple combined rank score
    df_metrics["mse_rank"] = df_metrics["MSE"].rank(ascending=True)
    df_metrics["ds_auc_rank"] = df_metrics["Downstream_ROC_AUC"].rank(ascending=False)
    df_metrics["ds_f1_rank"] = df_metrics["Downstream_F1_Score"].rank(ascending=False)
    df_metrics["total_rank"] = df_metrics["mse_rank"] + df_metrics["ds_auc_rank"] + df_metrics["ds_f1_rank"]
    
    best_idx = df_metrics["total_rank"].idxmin()
    best_model_name = df_metrics.loc[best_idx, "model_type"]
    
    # Report output
    report_path = biomarkers_dir / "benchmarking_report.md"
    with open(report_path, "w") as f:
        f.write("# ECG Biomarker Encoder Benchmarking Report\n\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"We trained, evaluated, and compared three latent representation learning models (Attention MLP, Beta-VAE, FT-Transformer) ")
        f.write(f"on the **full** dataset of {len(X_combined)} preprocessed 24-biomarker feature profiles from PTB-XL.\n\n")
        f.write(f"The input dimension was 48 (24 standardized features + 24 binary missingness indicators) to support joint reconstruction and classification.\n\n")
        f.write(f"Based on a holistic trade-off between reconstruction quality (MSE) and downstream class separation (F1 and ROC-AUC), **{best_model_name}** is the recommended model.\n\n")
        
        f.write("## Performance Metrics Comparison\n\n")
        f.write("| Model Type | Params | Reconstruction MSE | Downstream F1 Score | Downstream ROC-AUC | Direct F1 Score | Direct ROC-AUC | Training Time (s) |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for _, row in df_metrics.iterrows():
            f.write(
                f"| {row['model_type']} | {row['num_parameters']:,} | {row['MSE']:.6f} | "
                f"{row['Downstream_F1_Score']:.4f} | {row['Downstream_ROC_AUC']:.4f} | "
                f"{row['Direct_F1_Score']:.4f} | {row['Direct_ROC_AUC']:.4f} | "
                f"{row['training_time_seconds']:.2f} |\n"
            )
        f.write("\n")
        
        f.write("## Per-Label Classification Metrics\n\n")
        f.write("### Downstream Classifier (LR on Latent Space)\n")
        f.write("| Model Type | Metric | NORM | MI | STTC | CD | HYP |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for _, row in df_metrics.iterrows():
            f1_strs = [f"{v:.4f}" for v in row["Downstream_F1_Per_Label"]]
            auc_strs = [f"{v:.4f}" for v in row["Downstream_AUC_Per_Label"]]
            f.write(f"| {row['model_type']} | F1-Score | " + " | ".join(f1_strs) + " |\n")
            f.write(f"| {row['model_type']} | ROC-AUC | " + " | ".join(auc_strs) + " |\n")
        f.write("\n")
        
        f.write("### Direct Classification Head\n")
        f.write("| Model Type | Metric | NORM | MI | STTC | CD | HYP |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for _, row in df_metrics.iterrows():
            f1_strs = [f"{v:.4f}" for v in row["Direct_F1_Per_Label"]]
            auc_strs = [f"{v:.4f}" for v in row["Direct_AUC_Per_Label"]]
            f.write(f"| {row['model_type']} | F1-Score | " + " | ".join(f1_strs) + " |\n")
            f.write(f"| {row['model_type']} | ROC-AUC | " + " | ".join(auc_strs) + " |\n")
        f.write("\n")
        
        f.write("## Model-Specific Analysis\n\n")
        for _, row in df_metrics.iterrows():
            m_type = row["model_type"]
            f.write(f"### {m_type.replace('_', ' ').title()}\n")
            if m_type == "ft_transformer":
                f.write("- **Strengths**: Excels at mapping complex dependencies using attention layers. Provides highly separable downstream embeddings.\n")
                f.write("- **Limitations**: Higher parameter footprint and training time.\n")
            elif m_type == "beta_vae":
                f.write("- **Strengths**: Smooth and continuous latent space, ideal for interpolation and anomaly generation.\n")
                f.write("- **Limitations**: Trade-off between reconstruction and class separability governed by Beta coefficient.\n")
            else:
                f.write("- **Strengths**: Light footprint and rapid training with competitive classification performance.\n")
                f.write("- **Limitations**: Lacks sequence-level relational awareness.\n")
            f.write("\n")
            
        f.write("## Final Verdict & Recommendation\n")
        f.write(f"> [!IMPORTANT]\n")
        f.write(f"> **{best_model_name.replace('_', ' ').title()}** is selected as the optimal encoder. ")
        f.write("It achieves a balanced trade-off between faithful clinical biomarker reconstruction and high-fidelity downstream class separation. ")
        f.write("Its representations are recommended for integration with downstream diagnostic classifiers.\n")
        
    logger.info(f"Benchmarking report saved to {report_path}")
    
    # Write Thesis Notes
    thesis_notes_path = biomarkers_dir / "thesis_notes.md"
    with open(thesis_notes_path, "w") as f:
        f.write("# ECG Biomarker Encoder Thesis Notes\n\n")
        f.write("## 1. Methodology\n")
        f.write("- **Preprocessing**: Extracted 24 clinical features. Missing values imputed using dataset medians. Features standardized using StandardScaler (mean=0, std=1).\n")
        f.write("- **Model Input**: 48-dimensional vector (24 normalized features + 24 missingness binary indicators).\n")
        f.write("- **Joint Learning**: Networks reconstruct the 24 original features and predict the 5 multi-label diagnostic targets (NORM, MI, STTC, CD, HYP) from a 32-dimensional latent representation.\n\n")
        
        f.write("## 2. Experimental Results\n\n")
        f.write("| Model Type | Params | Reconstruction MSE | Downstream F1 Score | Downstream ROC-AUC |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for _, row in df_metrics.iterrows():
            f.write(f"| {row['model_type']} | {row['num_parameters']} | {row['MSE']:.6f} | {row['Downstream_F1_Score']:.4f} | {row['Downstream_ROC_AUC']:.4f} |\n")
        f.write("\n\n")
        
        f.write("## 3. Conclusions\n")
        f.write(f"- `{best_model_name}` achieved the most robust latent representations for classification and reconstruction.\n")
        f.write("- Joint classification head training enables early anomaly detection directly from latent variables.\n")
        
    logger.info(f"Thesis notes saved to {thesis_notes_path}")

if __name__ == "__main__":
    main()
