import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
# No seaborn dependency
pass
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score, 
    precision_recall_curve, accuracy_score, silhouette_score
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set up paths
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from biomarkers.models import AttentionMLPAutoencoder, BetaVAE, FTTransformerAutoencoder

# Settings
SEED = 42
LATENT_DIM = 32
BATCH_SIZE = 64

FEATURES = [
    "heart_rate", "mean_rr", "sd_rr", "p_amplitude", "p_duration", "pr_interval",
    "v1_r_amplitude", "v1_s_amplitude", "v5_r_amplitude", "max_r_v1_v6",
    "r_progression_slope", "max_st_elevation", "max_st_depression", "num_leads_st_deviation",
    "max_t_amplitude", "mean_t_amplitude", "num_leads_t_inversion", "qrs_duration",
    "qt_interval", "qtc_interval", "qrs_axis", "t_wave_axis", "qrs_t_angle", "sokolow_lyon"
]
LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]

def main():
    print("Starting Biomarker Encoder Pipeline Audit/Validation...")
    
    biomarkers_dir = project_root / "biomarkers"
    val_dir = biomarkers_dir / "validation"
    os.makedirs(val_dir, exist_ok=True)
    
    # ----------------------------------------------------
    # STAGE 1: EXTRACTION VALIDATION
    # ----------------------------------------------------
    print("\n--- STAGE 1: EXTRACTION VALIDATION ---")
    raw_csv = biomarkers_dir / "ecg_biomarkers_full.csv"
    log_csv = biomarkers_dir / "extraction_log_full.csv"
    
    if not raw_csv.exists():
        print(f"Error: {raw_csv} not found!")
        return
        
    df_raw = pd.read_csv(raw_csv)
    print(f"Loaded raw biomarkers. Shape: {df_raw.shape}")
    
    # 1. Expected features check
    missing_cols = [f for f in FEATURES if f not in df_raw.columns]
    if missing_cols:
        print(f"[STAGE 1 FAIL] Missing expected features: {missing_cols}")
    else:
        print("[STAGE 1 PASS] All 24 clinical features are present in CSV.")
        
    # 2. Compute descriptive statistics per feature
    stats_list = []
    for f in FEATURES:
        col = df_raw[f]
        missing_count = col.isna().sum()
        missing_pct = (missing_count / len(df_raw)) * 100.0
        
        # Outliers check (using 1.5 * IQR rule)
        q25 = col.quantile(0.25)
        q75 = col.quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = col[(col < lower_bound) | (col > upper_bound)]
        outlier_pct = (len(outliers) / len(df_raw)) * 100.0 if iqr > 0 else 0.0
        
        # Invalid values check (extreme physiological range violations)
        # e.g., heart rate <= 0 or > 350, negative duration/interval
        invalid_mask = pd.Series(False, index=df_raw.index)
        if f == "heart_rate":
            invalid_mask = (col <= 0) | (col > 350)
        elif f in ["p_duration", "pr_interval", "qrs_duration", "qt_interval", "qtc_interval", "mean_rr", "sd_rr", "sokolow_lyon"]:
            invalid_mask = col < 0
        elif f in ["qrs_axis", "t_wave_axis", "qrs_t_angle"]:
            invalid_mask = (col < -360) | (col > 360)
        invalid_count = invalid_mask.sum()
        invalid_pct = (invalid_count / len(df_raw)) * 100.0
        
        stats_list.append({
            "Feature": f,
            "Min": col.min(),
            "Max": col.max(),
            "Mean": col.mean(),
            "Median": col.median(),
            "Missing %": missing_pct,
            "Invalid %": invalid_pct,
            "Outlier %": outlier_pct
        })
        
    df_stats = pd.DataFrame(stats_list)
    stats_csv_path = val_dir / "extraction_feature_stats.csv"
    df_stats.to_csv(stats_csv_path, index=False)
    print(f"Saved feature statistics to {stats_csv_path}")
    
    # 3. Analyze warnings in logs
    if log_csv.exists():
        df_log = pd.read_csv(log_csv)
        print(f"Loaded extraction logs. Total rows: {len(df_log)}")
        
        warning_counts = {}
        for issue in df_log[df_log["status"] == "warning"]["issues"].dropna():
            parts = [w.strip() for w in issue.split(";") if w.strip()]
            for p in parts:
                warning_counts[p] = warning_counts.get(p, 0) + 1
                
        print("Warning counts from logs:")
        for w, cnt in sorted(warning_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {w}: {cnt} ({cnt/len(df_raw)*100.0:.2f}% of records)")
    else:
        print("Warning: extraction_log_full.csv not found.")
        
    # ----------------------------------------------------
    # STAGE 2: PREPROCESSING VALIDATION
    # ----------------------------------------------------
    print("\n--- STAGE 2: PREPROCESSING VALIDATION ---")
    print("[WARNING] Data leakage check: Imputer and Scaler are fit on the full dataset in run_experiments.py!")
    
    # Verify patient-wise train/validation/test splitting
    ptb_db_path = project_root / "data" / "raw" / "ptbxl" / "ptbxl_database.csv"
    if ptb_db_path.exists():
        df_ptb = pd.read_csv(ptb_db_path, index_col="ecg_id")
        patient_ids = df_ptb.loc[df_raw["record_id"], "patient_id"].values
        
        # Patient-wise split simulation
        unique_patients = np.unique(patient_ids)
        train_patients, test_patients = train_test_split(unique_patients, test_size=0.30, random_state=SEED)
        val_patients, test_patients = train_test_split(test_patients, test_size=0.50, random_state=SEED)
        
        train_idx = np.isin(patient_ids, train_patients)
        val_idx = np.isin(patient_ids, val_patients)
        test_idx = np.isin(patient_ids, test_patients)
        
        # Verify no overlap in patient_ids between splits
        train_p = set(patient_ids[train_idx])
        val_p = set(patient_ids[val_idx])
        test_p = set(patient_ids[test_idx])
        overlap_train_val = train_p.intersection(val_p)
        overlap_train_test = train_p.intersection(test_p)
        overlap_val_test = val_p.intersection(test_p)
        
        print(f"Patient IDs overlap check:")
        print(f"  - Train/Val overlap: {len(overlap_train_val)}")
        print(f"  - Train/Test overlap: {len(overlap_train_test)}")
        print(f"  - Val/Test overlap: {len(overlap_val_test)}")
        
        if len(overlap_train_val) + len(overlap_train_test) + len(overlap_val_test) == 0:
            print("[STAGE 2 PASS] Patient-wise splitting is technically correct and has zero patient overlap.")
        else:
            print("[STAGE 2 FAIL] Leakage detected: patient IDs overlap between splits!")
            
        # Class distribution and imbalance
        y = df_raw[LABELS].values
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
        
        print("\nClass distributions:")
        print(f"  Total records: {len(df_raw)}")
        print(f"  Train records: {len(y_train)} | Val records: {len(y_val)} | Test records: {len(y_test)}")
        for i, l in enumerate(LABELS):
            cnt_tr = y_train[:, i].sum()
            cnt_val = y_val[:, i].sum()
            cnt_te = y_test[:, i].sum()
            print(f"  - {l}: Train={cnt_tr} ({cnt_tr/len(y_train)*100.0:.1f}%), Val={cnt_val} ({cnt_val/len(y_val)*100.0:.1f}%), Test={cnt_te} ({cnt_te/len(y_test)*100.0:.1f}%)")
    else:
        print("Error: ptbxl_database.csv not found, cannot simulate patient-wise split.")
        
    # Check for row shifting or duplication
    if len(df_raw["record_id"].unique()) == len(df_raw):
        print("[STAGE 2 PASS] No duplicate record IDs found.")
    else:
        print("[STAGE 2 WARNING] Duplicate record IDs exist!")
        
    # ----------------------------------------------------
    # STAGE 3: BIOMARKER ENCODER VALIDATION
    # ----------------------------------------------------
    print("\n--- STAGE 3: BIOMARKER ENCODER VALIDATION ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Re-impute and scale correctly to check reconstruction and classification metrics
    X_raw_data = df_raw[FEATURES].values
    
    imputer_leakproof = SimpleImputer(strategy="median")
    scaler_leakproof = StandardScaler()
    
    X_train_raw = X_raw_data[train_idx]
    X_train_imputed = imputer_leakproof.fit_transform(X_train_raw)
    X_train_scaled = scaler_leakproof.fit_transform(X_train_imputed)
    
    X_val_imputed = imputer_leakproof.transform(X_raw_data[val_idx])
    X_val_scaled = scaler_leakproof.transform(X_val_imputed)
    
    X_test_imputed = imputer_leakproof.transform(X_raw_data[test_idx])
    X_test_scaled = scaler_leakproof.transform(X_test_imputed)
    
    M_train = (~pd.DataFrame(X_train_raw).isna()).astype(np.float32).values
    M_val = (~pd.DataFrame(X_raw_data[val_idx]).isna()).astype(np.float32).values
    M_test = (~pd.DataFrame(X_raw_data[test_idx]).isna()).astype(np.float32).values
    
    X_train_combined = np.hstack([X_train_scaled, M_train])
    X_val_combined = np.hstack([X_val_scaled, M_val])
    X_test_combined = np.hstack([X_test_scaled, M_test])
    
    models = {
        "attention_mlp": AttentionMLPAutoencoder(input_dim=48, latent_dim=LATENT_DIM, hidden_units=128, num_heads=4),
        "beta_vae": BetaVAE(input_dim=48, latent_dim=LATENT_DIM, hidden_units=128, beta=1.0),
        "ft_transformer": FTTransformerAutoencoder(input_dim=48, latent_dim=LATENT_DIM, d_model=32, nhead=2, num_layers=2, ffn_dim=64)
    }
    
    for name, model in models.items():
        ckpt_path = biomarkers_dir / f"{name}_best.pt"
        if not ckpt_path.exists():
            print(f"Skipping {name} (checkpoint not found at {ckpt_path})")
            continue
            
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Run on test set
        with torch.no_grad():
            test_x_tensor = torch.tensor(X_test_combined, dtype=torch.float32).to(device)
            if name == "beta_vae":
                recon, z, mu, logvar, class_logits = model(test_x_tensor)
                embeddings = mu.cpu().numpy()
            else:
                recon, latent, class_logits = model(test_x_tensor)
                embeddings = latent.cpu().numpy()
            recon = recon.cpu().numpy()
            class_logits = class_logits.cpu().numpy()
            
        # 1. Reconstruction metrics
        recon_mse = np.mean((X_test_scaled - recon) ** 2)
        recon_mae = np.mean(np.abs(X_test_scaled - recon))
        
        # 2. Embedding stats
        emb_mean = embeddings.mean(axis=0)
        emb_std = embeddings.std(axis=0)
        emb_min = embeddings.min(axis=0)
        emb_max = embeddings.max(axis=0)
        
        # Collapsed dimensions (std near 0)
        collapsed_dims = np.sum(emb_std < 0.01)
        total_var = np.var(embeddings, axis=0).sum()
        
        # Correlation matrix and redundancy
        corr_matrix = np.corrcoef(embeddings, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix)
        abs_corr = np.abs(corr_matrix)
        np.fill_diagonal(abs_corr, 0)
        mean_abs_corr = abs_corr.sum() / (LATENT_DIM * (LATENT_DIM - 1))
        high_corr_pairs = np.sum(abs_corr > 0.8) // 2
        
        # Check for NaN/Inf
        has_nan = np.isnan(embeddings).any()
        has_inf = np.isinf(embeddings).any()
        
        print(f"\nModel: {name.upper()}")
        print(f"  - Reconstruction MSE: {recon_mse:.6f}")
        print(f"  - Reconstruction MAE: {recon_mae:.6f}")
        print(f"  - Collapsed Dimensions (std < 0.01): {collapsed_dims} / {LATENT_DIM}")
        print(f"  - Total Embedding Variance: {total_var:.6f}")
        print(f"  - Mean Abs Correlation: {mean_abs_corr:.4f}")
        print(f"  - Highly Correlated Pairs (>0.8): {high_corr_pairs}")
        print(f"  - Has NaN/Inf: {has_nan} / {has_inf}")
        
        # Save embedding stats as CSV
        emb_df = pd.DataFrame({
            "dim": range(LATENT_DIM),
            "mean": emb_mean,
            "std": emb_std,
            "min": emb_min,
            "max": emb_max
        })
        emb_df.to_csv(val_dir / f"{name}_latent_stats.csv", index=False)
        
        # PCA
        pca = PCA(n_components=2, random_state=SEED)
        emb_pca = pca.fit_transform(embeddings)
        
        dominant_classes_test = []
        for row in y_test:
            active = [LABELS[i] for i in range(len(LABELS)) if row[i] == 1]
            dominant_classes_test.append(active[0] if active else "OTHER")
        dominant_classes_test = np.array(dominant_classes_test)
        
        # Silhouette Score
        valid_indices = dominant_classes_test != "OTHER"
        if valid_indices.sum() > 5:
            sil = silhouette_score(embeddings[valid_indices], dominant_classes_test[valid_indices])
            print(f"  - Silhouette Score (Dominant Class, clean): {sil:.4f}")
        else:
            sil = np.nan
            
        # Plot PCA
        plt.figure(figsize=(10, 8))
        unique_classes = sorted(list(set(dominant_classes_test)))
        for cls in unique_classes:
            idx = dominant_classes_test == cls
            plt.scatter(emb_pca[idx, 0], emb_pca[idx, 1], label=cls, alpha=0.7, edgecolors='none')
        plt.legend()
        plt.title(f"{name.upper()} PCA Space")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.tight_layout()
        plt.savefig(val_dir / f"{name}_pca.png")
        plt.close()
        
        # Plot t-SNE
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
        emb_tsne = tsne.fit_transform(embeddings[:1000])
        plt.figure(figsize=(10, 8))
        sub_dominant = dominant_classes_test[:1000]
        unique_classes_sub = sorted(list(set(sub_dominant)))
        for cls in unique_classes_sub:
            idx = sub_dominant == cls
            plt.scatter(emb_tsne[idx, 0], emb_tsne[idx, 1], label=cls, alpha=0.7, edgecolors='none')
        plt.legend()
        plt.title(f"{name.upper()} t-SNE Space (Subset of 1000)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(val_dir / f"{name}_tsne.png")
        plt.close()
        
        # Downstream Logistic Regression Classifier Evaluation (leakproof split)
        with torch.no_grad():
            train_x_tensor = torch.tensor(X_train_combined, dtype=torch.float32).to(device)
            if name == "beta_vae":
                _, _, train_mu, _, _ = model(train_x_tensor)
                train_embeddings = train_mu.cpu().numpy()
            else:
                _, train_latent, _ = model(train_x_tensor)
                train_embeddings = train_latent.cpu().numpy()
                
            val_x_tensor = torch.tensor(X_val_combined, dtype=torch.float32).to(device)
            if name == "beta_vae":
                _, _, val_mu, _, _ = model(val_x_tensor)
                val_embeddings = val_mu.cpu().numpy()
            else:
                _, val_latent, _ = model(val_x_tensor)
                val_embeddings = val_latent.cpu().numpy()
                
        ds_val_probs = np.zeros((len(X_val_combined), len(LABELS)))
        ds_test_probs = np.zeros((len(X_test_combined), len(LABELS)))
        
        for idx, label_name in enumerate(LABELS):
            clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
            clf.fit(train_embeddings, y_train[:, idx])
            ds_val_probs[:, idx] = clf.predict_proba(val_embeddings)[:, 1]
            ds_test_probs[:, idx] = clf.predict_proba(embeddings)[:, 1]
            
        thresholds = []
        for i in range(len(LABELS)):
            best_t = 0.5
            best_f1 = -1.0
            for t in np.linspace(0.01, 0.99, 99):
                preds = (ds_val_probs[:, i] >= t).astype(float)
                score = f1_score(y_val[:, i], preds, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_t = t
            thresholds.append(best_t)
            
        test_preds = np.zeros_like(ds_test_probs)
        for i in range(len(LABELS)):
            test_preds[:, i] = (ds_test_probs[:, i] >= thresholds[i]).astype(float)
            
        f1_macro = f1_score(y_test, test_preds, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, test_preds, average="weighted", zero_division=0)
        
        auc_roc_per_class = []
        auc_pr_per_class = []
        f1_per_class = []
        
        for i in range(len(LABELS)):
            auc_roc_per_class.append(roc_auc_score(y_test[:, i], ds_test_probs[:, i]))
            auc_pr_per_class.append(average_precision_score(y_test[:, i], ds_test_probs[:, i]))
            f1_per_class.append(f1_score(y_test[:, i], test_preds[:, i], zero_division=0))
            
        print(f"  - Downstream Macro F1: {f1_macro:.4f}")
        print(f"  - Downstream Weighted F1: {f1_weighted:.4f}")
        print(f"  - Downstream ROC-AUC Macro: {np.mean(auc_roc_per_class):.4f}")
        print(f"  - Downstream PR-AUC Macro: {np.mean(auc_pr_per_class):.4f}")
        for i, l in enumerate(LABELS):
            print(f"    * {l} - F1: {f1_per_class[i]:.4f} | ROC-AUC: {auc_roc_per_class[i]:.4f} | PR-AUC: {auc_pr_per_class[i]:.4f}")

    # ----------------------------------------------------
    # STAGE 4: END-TO-END PIPELINE & INFORMATION LOSS ANALYSIS
    # ----------------------------------------------------
    print("\n--- STAGE 4: END-TO-END PIPELINE & INFORMATION LOSS ANALYSIS ---")
    
    # Classifier A: Raw features (with median imputation)
    clf_a_probs = np.zeros((len(X_test_scaled), len(LABELS)))
    clf_a_val_probs = np.zeros((len(X_val_scaled), len(LABELS)))
    for idx in range(len(LABELS)):
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
        clf.fit(X_train_imputed, y_train[:, idx])
        clf_a_val_probs[:, idx] = clf.predict_proba(X_val_imputed)[:, 1]
        clf_a_probs[:, idx] = clf.predict_proba(X_test_imputed)[:, 1]
        
    thresh_a = []
    for i in range(len(LABELS)):
        best_t = 0.5
        best_f1 = -1.0
        for t in np.linspace(0.01, 0.99, 99):
            preds = (clf_a_val_probs[:, i] >= t).astype(float)
            score = f1_score(y_val[:, i], preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = t
        thresh_a.append(best_t)
        
    preds_a = np.zeros_like(clf_a_probs)
    for i in range(len(LABELS)):
        preds_a[:, i] = (clf_a_probs[:, i] >= thresh_a[i]).astype(float)
        
    f1_a = f1_score(y_test, preds_a, average="macro", zero_division=0)
    roc_a = roc_auc_score(y_test, clf_a_probs, average="macro")
    pr_a = average_precision_score(y_test, clf_a_probs, average="macro")
    
    # Classifier B: Preprocessed scaled features
    clf_b_probs = np.zeros((len(X_test_scaled), len(LABELS)))
    clf_b_val_probs = np.zeros((len(X_val_scaled), len(LABELS)))
    for idx in range(len(LABELS)):
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
        clf.fit(X_train_scaled, y_train[:, idx])
        clf_b_val_probs[:, idx] = clf.predict_proba(X_val_scaled)[:, 1]
        clf_b_probs[:, idx] = clf.predict_proba(X_test_scaled)[:, 1]
        
    thresh_b = []
    for i in range(len(LABELS)):
        best_t = 0.5
        best_f1 = -1.0
        for t in np.linspace(0.01, 0.99, 99):
            preds = (clf_b_val_probs[:, i] >= t).astype(float)
            score = f1_score(y_val[:, i], preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = t
        thresh_b.append(best_t)
        
    preds_b = np.zeros_like(clf_b_probs)
    for i in range(len(LABELS)):
        preds_b[:, i] = (clf_b_probs[:, i] >= thresh_b[i]).astype(float)
        
    f1_b = f1_score(y_test, preds_b, average="macro", zero_division=0)
    roc_b = roc_auc_score(y_test, clf_b_probs, average="macro")
    pr_b = average_precision_score(y_test, clf_b_probs, average="macro")
    
    # Classifier C: Best embeddings (Attention MLP)
    model = AttentionMLPAutoencoder(input_dim=48, latent_dim=LATENT_DIM, hidden_units=128, num_heads=4)
    model.load_state_dict(torch.load(biomarkers_dir / "attention_mlp_best.pt", map_location=device))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        tr_tensor = torch.tensor(X_train_combined, dtype=torch.float32).to(device)
        te_tensor = torch.tensor(X_test_combined, dtype=torch.float32).to(device)
        val_tensor = torch.tensor(X_val_combined, dtype=torch.float32).to(device)
        
        _, train_emb_c, _ = model(tr_tensor)
        _, test_emb_c, _ = model(te_tensor)
        _, val_emb_c, _ = model(val_tensor)
        
        train_emb_c = train_emb_c.cpu().numpy()
        test_emb_c = test_emb_c.cpu().numpy()
        val_emb_c = val_emb_c.cpu().numpy()
        
    clf_c_probs = np.zeros((len(X_test_scaled), len(LABELS)))
    clf_c_val_probs = np.zeros((len(X_val_scaled), len(LABELS)))
    for idx in range(len(LABELS)):
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
        clf.fit(train_emb_c, y_train[:, idx])
        clf_c_val_probs[:, idx] = clf.predict_proba(val_emb_c)[:, 1]
        clf_c_probs[:, idx] = clf.predict_proba(test_emb_c)[:, 1]
        
    thresh_c = []
    for i in range(len(LABELS)):
        best_t = 0.5
        best_f1 = -1.0
        for t in np.linspace(0.01, 0.99, 99):
            preds = (clf_c_val_probs[:, i] >= t).astype(float)
            score = f1_score(y_val[:, i], preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = t
        thresh_c.append(best_t)
        
    preds_c = np.zeros_like(clf_c_probs)
    for i in range(len(LABELS)):
        preds_c[:, i] = (clf_c_probs[:, i] >= thresh_c[i]).astype(float)
        
    f1_c = f1_score(y_test, preds_c, average="macro", zero_division=0)
    roc_c = roc_auc_score(y_test, clf_c_probs, average="macro")
    pr_c = average_precision_score(y_test, clf_c_probs, average="macro")
    
    print("\nInformation Loss Comparison (Linear Classifier on Frozen Reps):")
    print(f"  - A. Raw features (Imputed)      -> Macro F1: {f1_a:.4f} | ROC-AUC: {roc_a:.4f} | PR-AUC: {pr_a:.4f}")
    print(f"  - B. Preprocessed (Imputed+Scaled) -> Macro F1: {f1_b:.4f} | ROC-AUC: {roc_b:.4f} | PR-AUC: {pr_b:.4f}")
    print(f"  - C. 32-D Attention MLP Embedding  -> Macro F1: {f1_c:.4f} | ROC-AUC: {roc_c:.4f} | PR-AUC: {pr_c:.4f}")
    
    # Trace 5 representative test records
    print("\nTracing 5 representative test records through the end-to-end pipeline:")
    rep_records = []
    for idx, l in enumerate(LABELS):
        pos_idx = np.where(y_test[:, idx] == 1)[0]
        if len(pos_idx) > 0:
            rep_records.append((pos_idx[0], l))
            
    for test_idx_sub, label_name in rep_records[:5]:
        idx_in_raw = np.where(test_idx)[0][test_idx_sub]
        rec_id = df_raw.iloc[idx_in_raw]["record_id"]
        
        # Raw features
        idx_in_raw = np.where(test_idx)[0][test_idx_sub]
        raw_feats = df_raw.iloc[idx_in_raw][FEATURES].to_dict()
        
        # Preprocessed (Imputed & Scaled)
        preprocessed_feats = X_test_scaled[test_idx_sub]
        
        # Embedding
        emb = test_emb_c[test_idx_sub]
        
        # Predictions (Probabilities)
        pred_probs = clf_c_probs[test_idx_sub]
        true_labels = y_test[test_idx_sub]
        
        print(f"\nRecord ID: {rec_id} (True Dominant/Selected Class: {label_name})")
        print(f"  Raw features (first 5): {list(raw_feats.items())[:5]}")
        print(f"  Missing features: {[f for f, v in raw_feats.items() if pd.isna(v)]}")
        print(f"  Preprocessed features (first 5): {list(preprocessed_feats)[:5]}")
        print(f"  32-D Embedding (first 5): {list(emb)[:5]}")
        print(f"  True Label Vector: {dict(zip(LABELS, true_labels))}")
        print(f"  Predicted Probabilities: {dict(zip(LABELS, pred_probs))}")
        
    print("\nValidation complete!")

if __name__ == "__main__":
    main()
