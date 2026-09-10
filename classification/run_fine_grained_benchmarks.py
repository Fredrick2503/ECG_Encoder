import os
import sys
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import mlflow
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc, brier_score_loss
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from data_management.dataset_factory import DatasetFactory
from classification.subclass_encoder import PTBXLSubclassLabelEncoder, PTBXL_SUBCLASSES
from classification.classifier import ZFusedDataset, LinearProbeClassifier, MLPClassifier
from classification.losses import BCEWithLogitsLoss
from classification.metrics import compute_binary_ece

BATCH_SIZE = 64
EPOCHS = 30
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def calculate_pr_auc(y_true, y_prob):
    """Computes Precision-Recall Area Under Curve (PR-AUC)."""
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        return float(auc(recall, precision))
    except Exception:
        return np.nan

def optimize_thresholds_f1(labels, probs):
    """Standard per-class F1-maximizing threshold grid search."""
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
        thrs = optimize_thresholds_f1(val_labels, val_probs)
        preds = (val_probs >= thrs).astype(int)
        val_f1 = f1_score(val_labels, preds, average='macro', zero_division=0)
        
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

def compute_knn_purity_subclass(feats, labels, k=5):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, metric="cosine")
    nbrs.fit(feats)
    _, indices = nbrs.kneighbors(feats)
    purities = []
    for i in range(len(feats)):
        nn_idx = indices[i, 1:] # exclude self
        self_labels = labels[i]
        nn_labels = labels[nn_idx]
        inter = np.minimum(self_labels, nn_labels).sum(axis=1)
        union = np.maximum(self_labels, nn_labels).sum(axis=1)
        jaccard = np.where(union > 0, inter / union, 1.0)
        purities.append(jaccard.mean())
    return float(np.mean(purities))

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow_benchmark.db")
    mlflow.set_experiment("ECG_Fine_Grained_Classification")
    
    print("=" * 70)
    print("Fine-Grained Diagnostic Subclass Validation Suite")
    print("=" * 70)
    
    # 1. Load data factory & metadata mappings
    print("Loading PTB-XL metadata & splits...")
    train_ds, val_ds, test_ds, loader = DatasetFactory.create_datasets(
        dataset_type="ptbxl", download=False, resolution="lr"
    )
    metadata_df = loader.metadata_df
    
    # 2. Load pre-extracted representations
    data_file = project_root / "data" / "Z_fused_2k.npz"
    data = np.load(data_file)
    
    train_z = data["train_z_fused"]
    val_z = data["val_z_fused"]
    test_z = data["test_z_fused"]
    
    train_record_ids = data["train_record_id"]
    val_record_ids = data["val_record_id"]
    test_record_ids = data["test_record_id"]
    
    # 3. Encode subclasses ground truth target vectors
    sub_encoder = PTBXLSubclassLabelEncoder()
    num_classes = len(PTBXL_SUBCLASSES)
    
    def extract_subclass_labels(record_ids):
        labels_list = []
        for rid in record_ids:
            row = metadata_df.loc[int(rid)]
            scp_codes = row.get("scp_codes", {})
            subs = loader.parser.get_subclasses(scp_codes)
            labels_list.append(sub_encoder.encode(subs))
        return np.array(labels_list)
        
    print("Encoding target subclasses multi-hot vectors...")
    train_y_sub = extract_subclass_labels(train_record_ids)
    val_y_sub = extract_subclass_labels(val_record_ids)
    test_y_sub = extract_subclass_labels(test_record_ids)
    
    class_counts = train_y_sub.sum(axis=0)
    print(f"Total training subclasses extracted (Support per subclass):")
    for s_name, count in zip(PTBXL_SUBCLASSES, class_counts):
        print(f"  {s_name}: {int(count)}")
        
    train_loader = DataLoader(ZFusedDataset(train_z, train_y_sub), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ZFusedDataset(val_z, val_y_sub), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ZFusedDataset(test_z, test_y_sub), batch_size=BATCH_SIZE, shuffle=False)
    
    # ─── C0 BASELINE: Linear Probe on Subclasses ──────────────────────────────
    print("\n--- Training C0 Linear Probe (Fine Labels) ---")
    model_c0 = LinearProbeClassifier(input_dim=1056, num_classes=num_classes)
    loss_fn = BCEWithLogitsLoss()
    model_c0 = train_model(model_c0, train_loader, val_loader, loss_fn)
    
    val_probs_c0, _ = evaluate_model(model_c0, val_loader)
    thrs_c0 = optimize_thresholds_f1(val_y_sub, val_probs_c0)
    
    test_probs_c0, _ = evaluate_model(model_c0, test_loader)
    preds_c0 = (test_probs_c0 >= thrs_c0).astype(int)
    
    macro_f1_c0 = f1_score(test_y_sub, preds_c0, average='macro', zero_division=0)
    macro_auc_c0 = roc_auc_score(test_y_sub, test_probs_c0, average='macro')
    print(f"C0 Linear Probe Test: Macro F1 = {macro_f1_c0:.4f} | Macro AUC = {macro_auc_c0:.4f}")
    
    # ─── C1 MLP Model ─────────────────────────────────────────────────────────
    print("\n--- Training C1 MLP (Fine Labels) ---")
    model_c1 = MLPClassifier(input_dim=1056, hidden_dim=256, num_classes=num_classes)
    model_c1 = train_model(model_c1, train_loader, val_loader, loss_fn)
    
    val_probs_c1, _ = evaluate_model(model_c1, val_loader)
    thrs_c1 = optimize_thresholds_f1(val_y_sub, val_probs_c1)
    
    test_probs_c1, _ = evaluate_model(model_c1, test_loader)
    preds_c1 = (test_probs_c1 >= thrs_c1).astype(int)
    
    macro_f1_c1 = f1_score(test_y_sub, preds_c1, average='macro', zero_division=0)
    macro_auc_c1 = roc_auc_score(test_y_sub, test_probs_c1, average='macro')
    print(f"C1 MLP Test: Macro F1 = {macro_f1_c1:.4f} | Macro AUC = {macro_auc_c1:.4f}")
    
    # Detailed diagnosis-level metrics
    diag_metrics = []
    for c in range(num_classes):
        y_true = test_y_sub[:, c]
        y_prob = test_probs_c1[:, c]
        y_pred = preds_c1[:, c]
        
        auc_score = roc_auc_score(y_true, y_prob) if y_true.sum() > 0 else np.nan
        pr_auc = calculate_pr_auc(y_true, y_prob)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Sensitivity / Specificity
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        
        # Calibration
        ece = compute_binary_ece(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        
        diag_metrics.append({
            "Class": PTBXL_SUBCLASSES[c],
            "F1-Score": f1,
            "ROC-AUC": auc_score,
            "PR-AUC": pr_auc,
            "Sensitivity": sens,
            "Specificity": spec,
            "ECE": ece,
            "Brier": brier,
            "Support": int(y_true.sum()),
            "Train_Support": int(class_counts[c])
        })
    df_diag = pd.DataFrame(diag_metrics)
    
    # ─── RARE CLASS ANALYSIS ──────────────────────────────────────────────────
    print("\n--- Rare Class Analysis ---")
    rare_mask = df_diag["Train_Support"] <= 15
    med_mask = (df_diag["Train_Support"] > 15) & (df_diag["Train_Support"] <= 100)
    freq_mask = df_diag["Train_Support"] > 100
    
    rare_f1 = df_diag[rare_mask]["F1-Score"].mean()
    med_f1 = df_diag[med_mask]["F1-Score"].mean()
    freq_f1 = df_diag[freq_mask]["F1-Score"].mean()
    
    print(f"  Rare (Support <= 15) Mean F1:   {rare_f1:.4f} (N={rare_mask.sum()})")
    print(f"  Medium (15 < Support <= 100) Mean F1: {med_f1:.4f} (N={med_mask.sum()})")
    print(f"  Frequent (Support > 100) Mean F1: {freq_f1:.4f} (N={freq_mask.sum()})")
    
    # ─── PAIRWISE CLINICAL SUBTYPE DISCRIMINATION ─────────────────────────────
    print("\n--- Pairwise Subclass Discrimination Analysis ---")
    # 1. Myocardial Infarction subtypes
    mi_classes = ["IMI", "AMI", "LMI", "PMI"]
    mi_corr = np.corrcoef(test_probs_c1[:, [PTBXL_SUBCLASSES.index(cls) for cls in mi_classes]].T)
    print("  MI Subtypes Correlation Matrix:")
    print(pd.DataFrame(mi_corr, columns=mi_classes, index=mi_classes).to_string())
    
    # 2. Conduction Defects & Bundle Branch Blocks
    block_classes = ["CRBBB", "CLBBB", "IRBBB", "ILBBB"]
    block_corr = np.corrcoef(test_probs_c1[:, [PTBXL_SUBCLASSES.index(cls) for cls in block_classes]].T)
    print("\n  Block Subtypes Correlation Matrix:")
    print(pd.DataFrame(block_corr, columns=block_classes, index=block_classes).to_string())

    # ─── MODALITY COMPLEMENTARITY ANALYSIS (T vs T+M vs T+M+B) ────────────────
    print("\n--- Modality Complementarity Study on Subclasses ---")
    ablation_sets = {
        "T (Temporal)": (0, 512),
        "T+M (Pairwise)": (0, 1024),
        "T+M+B (Full Fused)": (0, 1056)
    }
    
    ablation_class_f1 = {}
    
    for name, indices in ablation_sets.items():
        start, end = indices
        trn_sub = train_z[:, start:end]
        val_sub = val_z[:, start:end]
        tst_sub = test_z[:, start:end]
        
        sub_loader_trn = DataLoader(ZFusedDataset(trn_sub, train_y_sub), batch_size=BATCH_SIZE, shuffle=True)
        sub_loader_val = DataLoader(ZFusedDataset(val_sub, val_y_sub), batch_size=BATCH_SIZE, shuffle=False)
        sub_loader_tst = DataLoader(ZFusedDataset(tst_sub, test_y_sub), batch_size=BATCH_SIZE, shuffle=False)
        
        probe_sub = LinearProbeClassifier(input_dim=trn_sub.shape[1], num_classes=num_classes)
        probe_sub = train_model(probe_sub, sub_loader_trn, sub_loader_val, loss_fn)
        
        val_probs_sub, _ = evaluate_model(probe_sub, sub_loader_val)
        thrs_sub = optimize_thresholds_f1(val_y_sub, val_probs_sub)
        
        tst_probs_sub, _ = evaluate_model(probe_sub, sub_loader_tst)
        preds_sub = (tst_probs_sub >= thrs_sub).astype(int)
        
        # Calculate per-class F1 for this slice
        ablation_class_f1[name] = [f1_score(test_y_sub[:, c], preds_sub[:, c], zero_division=0) for c in range(num_classes)]
        
    df_comp = pd.DataFrame(ablation_class_f1, index=PTBXL_SUBCLASSES)
    
    # Highlight biomarker contributions on rare/specific diagnoses
    print("\n  Detailed Modality Comparison Table:")
    print(df_comp.to_string())
    
    # ─── EMBEDDING SEPARABILITY ANALYSIS ──────────────────────────────────────
    print("\n--- Embedding Separability metrics ---")
    test_zf_norm = test_z / (np.linalg.norm(test_z, axis=1, keepdims=True) + 1e-8)
    
    # Cosine kNN Purity
    purity = compute_knn_purity_subclass(test_zf_norm, test_y_sub, k=5)
    
    # Clustering metrics with K-Means (K=5 matching major diagnostic axes)
    km = KMeans(n_clusters=5, random_state=SEED, n_init=25)
    clusters = km.fit_predict(test_zf_norm)
    
    test_dom_subclass = np.argmax(test_y_sub, axis=1)
    sil = silhouette_score(test_zf_norm, clusters)
    nmi = normalized_mutual_info_score(test_dom_subclass, clusters)
    ari = adjusted_rand_score(test_dom_subclass, clusters)
    
    print(f"  Subclass Neighborhood Purity@5: {purity:.4f}")
    print(f"  Subclass Clustering Silhouette:  {sil:.4f}")
    print(f"  Subclass Clustering NMI vs GT:   {nmi:.4f}")
    print(f"  Subclass Clustering ARI vs GT:   {ari:.4f}")

    # ─── SAVE REPORTS ─────────────────────────────────────────────────────────
    report_dir = project_root / "outputs/reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = report_dir / "fine_grained_validation_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 4: Fine-Grained Diagnostic Validation Report\n\n")
        f.write("This benchmark study evaluates whether the joint fused representation space retains detailed clinical subclasses.\n\n")
        
        f.write("## 1. Subclass Diagnostic Performance Comparison (C1 MLP)\n\n")
        f.write(df_diag.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 2. Rare Class Performance Summary\n\n")
        f.write(f"- **Rare Classes (Support <= 15)**: Mean F1 = `{rare_f1:.4f}`\n")
        f.write(f"- **Medium Classes (15 < Support <= 100)**: Mean F1 = `{med_f1:.4f}`\n")
        f.write(f"- **Frequent Classes (Support > 100)**: Mean F1 = `{freq_f1:.4f}`\n\n")
        
        f.write("## 3. Subclass Embedding Separability\n\n")
        f.write(f"- **Cosine kNN Purity (k=5)**: `{purity:.4f}`\n")
        f.write(f"- **K-Means Silhouette (k=5)**: `{sil:.4f}`\n")
        f.write(f"- **Clustering NMI vs Dominant Subclass**: `{nmi:.4f}`\n")
        f.write(f"- **Clustering ARI vs Dominant Subclass**: `{ari:.4f}`\n\n")
        
        f.write("## 4. Modality Complementarity Table (Per-Class F1 comparison)\n\n")
        f.write(df_comp.to_markdown())
        f.write("\n\n")
        
        f.write("## 5. Pairwise Correlation Matrix\n\n")
        f.write("### Myocardial Infarction Subtypes\n\n")
        f.write(pd.DataFrame(mi_corr, columns=mi_classes, index=mi_classes).to_markdown())
        f.write("\n\n")
        f.write("### Conduction Blocks & Delay Subtypes\n\n")
        f.write(pd.DataFrame(block_corr, columns=block_classes, index=block_classes).to_markdown())
        f.write("\n")
        
    print(f"Saved report to {report_path}")
    
    # Log runs to MLflow
    with mlflow.start_run(run_name="Fine_Grained_Validation"):
        mlflow.log_metric("C0_subclass_macro_f1", macro_f1_c0)
        mlflow.log_metric("C0_subclass_macro_auc", macro_auc_c0)
        mlflow.log_metric("C1_subclass_macro_f1", macro_f1_c1)
        mlflow.log_metric("C1_subclass_macro_auc", macro_auc_c1)
        mlflow.log_metric("Subclass_Purity_k5", purity)
        mlflow.log_metric("Subclass_NMI", nmi)

if __name__ == "__main__":
    main()
