import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from biomarkers.models import AttentionMLPAutoencoder
from explainability.shap import SHAPExplainerWrapper

# Constants
SEED = 42
FEATURES = [
    "heart_rate", "mean_rr", "sd_rr", "p_amplitude", "p_duration", "pr_interval",
    "v1_r_amplitude", "v1_s_amplitude", "v5_r_amplitude", "max_r_v1_v6",
    "r_progression_slope", "max_st_elevation", "max_st_depression", "num_leads_st_deviation",
    "max_t_amplitude", "mean_t_amplitude", "num_leads_t_inversion", "qrs_duration",
    "qt_interval", "qtc_interval", "qrs_axis", "t_wave_axis", "qrs_t_angle", "sokolow_lyon"
]
LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]

class ECGFeatureDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.X[idx]

def load_data_and_fit_downstream(biomarkers_dir, device):
    """
    Loads full dataset, splits it patient-wise, and fits downstream classifiers on the fly.
    """
    raw_csv = biomarkers_dir / "ecg_biomarkers_full.csv"
    print(f"Loading raw features from {raw_csv} for fitting downstream classifiers...")
    df_raw = pd.read_csv(raw_csv)
    
    # Preprocess
    with open(biomarkers_dir / "imputer.pkl", "rb") as f:
        imputer = pickle.load(f)
    with open(biomarkers_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        
    X_imputed = imputer.transform(df_raw[FEATURES])
    X_scaled = scaler.transform(X_imputed)
    M = (~df_raw[FEATURES].isna()).astype(np.float32).values
    X_combined = np.hstack([X_scaled, M])
    y = df_raw[LABELS].values
    
    # Split patient-wise
    ptb_db_path = project_root / "data" / "raw" / "ptbxl" / "ptbxl_database.csv"
    if ptb_db_path.exists():
        df_ptb = pd.read_csv(ptb_db_path, index_col="ecg_id")
        patient_ids = df_ptb.loc[df_raw["record_id"], "patient_id"].values
        unique_patients = np.unique(patient_ids)
        train_patients, test_patients = train_test_split(unique_patients, test_size=0.30, random_state=SEED)
        val_patients, test_patients = train_test_split(test_patients, test_size=0.50, random_state=SEED)
        
        train_idx = np.isin(patient_ids, train_patients)
        val_idx = np.isin(patient_ids, val_patients)
        test_idx = np.isin(patient_ids, test_patients)
        
        X_train_raw = df_raw.loc[train_idx, FEATURES].values
        X_train = X_combined[train_idx]
        y_train = y[train_idx]
        
        X_val = X_combined[val_idx]
        y_val = y[val_idx]
        
        X_test_raw = df_raw.loc[test_idx, FEATURES].values
        y_test = y[test_idx]
        test_record_ids = df_raw.loc[test_idx, "record_id"].values
    else:
        print("ptbxl_database.csv not found. Falling back to record-wise split.")
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(df_raw[FEATURES].values, y, test_size=0.30, random_state=SEED)
        X_train, X_test, _, _ = train_test_split(X_combined, y, test_size=0.30, random_state=SEED)
        # Placeholder val split
        X_val, y_val = X_train, y_train
        test_record_ids = np.arange(len(X_test))
        
    # Load model
    input_dim = X_combined.shape[1]
    model = AttentionMLPAutoencoder(input_dim=input_dim, latent_dim=32, hidden_units=128, num_heads=4)
    checkpoint_path = biomarkers_dir / "attention_mlp_best.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Generate train embeddings
    print("Generating train embeddings...")
    train_loader = DataLoader(ECGFeatureDataset(X_train), batch_size=256, shuffle=False)
    train_embs = []
    with torch.no_grad():
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            _, latent, _ = model(batch_x)
            train_embs.append(latent.cpu().numpy())
    train_embs = np.concatenate(train_embs, axis=0)
    
    # Fit downstream Logistic Regression classifiers
    print("Fitting class-weighted downstream classifiers...")
    classifiers = []
    for idx, label_name in enumerate(LABELS):
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
        clf.fit(train_embs, y_train[:, idx])
        classifiers.append(clf)
        
    return model, imputer, scaler, classifiers, X_train_raw, X_val, y_val, X_test_raw, y_test, test_record_ids

def tune_thresholds(model, classifiers, X_val, y_val, device):
    """
    Computes optimal F1 thresholds on the validation set for each diagnostic class.
    """
    print("Tuning decision thresholds on validation set...")
    val_loader = DataLoader(ECGFeatureDataset(X_val), batch_size=256, shuffle=False)
    val_embs = []
    model.eval()
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            _, latent, _ = model(batch_x)
            val_embs.append(latent.cpu().numpy())
    val_embs = np.concatenate(val_embs, axis=0)
    
    val_probs = np.zeros((len(X_val), len(LABELS)))
    for idx in range(len(LABELS)):
        val_probs[:, idx] = classifiers[idx].predict_proba(val_embs)[:, 1]
        
    thresholds = np.zeros(len(LABELS))
    for i in range(len(LABELS)):
        best_t = 0.5
        best_f1 = -1.0
        for t in np.linspace(0.01, 0.99, 99):
            preds = (val_probs[:, i] >= t).astype(float)
            score = f1_score(y_val[:, i], preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = t
        thresholds[i] = best_t
        print(f"  - {LABELS[i]}: threshold = {best_t:.4f} (best F1 = {best_f1:.4f})")
    return thresholds

def generate_html_report(reports_data, thresholds, save_dir):
    """
    Generates a premium HTML page to display SHAP explanation plots and predictions.
    """
    html_path = Path(save_dir) / "index.html"
    
    # Generate cards for each record
    cards_html = ""
    for r in reports_data:
        record_id = r["record_id"]
        true_labels = ", ".join(r["true_labels"]) if r["true_labels"] else "None"
        
        # Build probability pills
        prob_pills = ""
        for i, lbl in enumerate(LABELS):
            prob = r["probs"][i]
            thresh = thresholds[i]
            is_pred = prob >= thresh
            pill_class = "pill-active" if is_pred else "pill-inactive"
            prob_pills += f'<div class="pill {pill_class}">{lbl}: {prob:.4f} (thresh: {thresh:.2f})</div>'
            
        # Explanations list
        exp_sections = ""
        for exp in r["explanations"]:
            class_name = exp["class_name"]
            prob = exp["prob"]
            img_filename = exp["img_filename"]
            
            pos_contribs = "".join([f'<li><strong>{feat}</strong>: +{val:.4f} (val: {val_str})</li>' for feat, val, val_str in exp["top_pos"]])
            neg_contribs = "".join([f'<li><strong>{feat}</strong>: {val:.4f} (val: {val_str})</li>' for feat, val, val_str in exp["top_neg"]])
            
            exp_sections += f"""
            <div class="explanation-block">
                <div class="exp-header">SHAP Explanation for prediction: {class_name} ({prob:.4f})</div>
                <div class="exp-content">
                    <div class="exp-metrics">
                        <h5>Top Positive Contributing Biomarkers</h5>
                        <ul>{pos_contribs if pos_contribs else "<li>None</li>"}</ul>
                        <h5>Top Negative Contributing Biomarkers</h5>
                        <ul>{neg_contribs if neg_contribs else "<li>None</li>"}</ul>
                    </div>
                    <div class="exp-plot">
                        <img src="{img_filename}" alt="SHAP Plot {class_name}" class="shap-img">
                    </div>
                </div>
            </div>
            """
            
        cards_html += f"""
        <div class="card">
            <div class="card-header">
                <h3>Record ID: {record_id}</h3>
                <span class="gt-badge">Ground Truth: {true_labels}</span>
            </div>
            <div class="card-body">
                <div class="probs-section">
                    <h4>Pipeline Probabilities</h4>
                    <div class="pills-container">{prob_pills}</div>
                </div>
                {exp_sections}
            </div>
        </div>
        """
        
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ECG Biomarker SHAP Explanations Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: #151c2c;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --primary: #6366f1;
            --accent-pos: #ff0d57;
            --accent-neg: #1e88e5;
            --border-color: #243049;
        }}
        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            text-align: center;
            margin-bottom: 50px;
        }}
        h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            background: linear-gradient(to right, #6366f1, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        header p {{
            color: var(--text-secondary);
            font-size: 1.1rem;
        }}
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            margin-bottom: 40px;
            overflow: hidden;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3);
        }}
        .card-header {{
            padding: 20px;
            background-color: rgba(99, 102, 241, 0.08);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .card-header h3 {{
            margin: 0;
            font-size: 1.4rem;
        }}
        .gt-badge {{
            background-color: #374151;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            color: #e5e7eb;
            font-weight: 600;
        }}
        .card-body {{
            padding: 24px;
        }}
        .probs-section {{
            margin-bottom: 30px;
        }}
        .probs-section h4 {{
            margin-top: 0;
            margin-bottom: 15px;
            color: var(--text-secondary);
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .pills-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .pill {{
            padding: 8px 16px;
            border-radius: 30px;
            font-size: 0.9rem;
            font-weight: 600;
        }}
        .pill-active {{
            background-color: rgba(99, 102, 241, 0.2);
            color: #818cf8;
            border: 1px solid rgba(99, 102, 241, 0.4);
        }}
        .pill-inactive {{
            background-color: rgba(255, 255, 255, 0.03);
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
        }}
        .explanation-block {{
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background-color: rgba(0,0,0,0.15);
            margin-bottom: 24px;
            overflow: hidden;
        }}
        .explanation-block:last-child {{
            margin-bottom: 0;
        }}
        .exp-header {{
            padding: 12px 20px;
            background-color: rgba(255, 255, 255, 0.02);
            border-bottom: 1px solid var(--border-color);
            font-weight: 600;
            color: #818cf8;
        }}
        .exp-content {{
            display: flex;
            flex-direction: row;
            flex-wrap: wrap;
            padding: 20px;
            gap: 20px;
        }}
        .exp-metrics {{
            flex: 1;
            min-width: 300px;
        }}
        .exp-metrics h5 {{
            margin-top: 0;
            margin-bottom: 10px;
            font-size: 0.95rem;
            font-weight: 600;
        }}
        .exp-metrics h5:nth-of-type(1) {{
            color: var(--accent-pos);
        }}
        .exp-metrics h5:nth-of-type(2) {{
            color: var(--accent-neg);
        }}
        .exp-metrics ul {{
            margin: 0 0 20px 0;
            padding-left: 20px;
            color: #d1d5db;
            font-size: 0.9rem;
        }}
        .exp-metrics li {{
            margin-bottom: 6px;
        }}
        .exp-plot {{
            flex: 1.5;
            min-width: 400px;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        .shap-img {{
            max-width: 100%;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>ECG Biomarker SHAP Explanations Report</h1>
            <p>Instance-level feature attributions for multi-label predictions across the Attention MLP pipeline</p>
        </header>
        <main>
            {cards_html}
        </main>
    </div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\nHTML report created successfully at: {html_path}")
    print(f"You can view it directly in your browser: file:///{html_path.resolve().as_posix()}")

def main():
    parser = argparse.ArgumentParser(description="Generate instance-level SHAP explanations for ECG biomarker predictions.")
    parser.add_argument("--record-id", type=int, default=None, help="PTB-XL record ID to fetch and explain (requires full dataset).")
    parser.add_argument("--test-index", type=int, default=None, help="Index inside the test set split to explain.")
    parser.add_argument("--save-dir", type=str, default="outputs/shap_explanations", help="Directory to save SHAP plots.")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    biomarkers_dir = project_root / "biomarkers"
    
    # Load model and fit classifiers
    model, imputer, scaler, classifiers, X_train_raw, X_val, y_val, X_test_raw, y_test, test_record_ids = load_data_and_fit_downstream(biomarkers_dir, device)
    
    # Tune decision thresholds on validation set
    thresholds = tune_thresholds(model, classifiers, X_val, y_val, device)
    
    # Setup Explainer Wrapper
    print("\nInitializing SHAP KernelExplainer...")
    explainer_wrapper = SHAPExplainerWrapper(model, imputer, scaler, classifiers, FEATURES, LABELS, device)
    # Fit explainer on background training sample
    explainer_wrapper.fit_explainer(X_train_raw, n_samples=50)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Select which records to explain
    selected_indices = []
    if args.record_id is not None:
        match_idx = np.where(test_record_ids == args.record_id)[0]
        if len(match_idx) > 0:
            selected_indices = [match_idx[0]]
        else:
            print(f"Error: Record ID {args.record_id} not found in the test split.")
            sys.exit(1)
    elif args.test_index is not None:
        idx = args.test_index
        if 0 <= idx < len(X_test_raw):
            selected_indices = [idx]
        else:
            print(f"Error: Invalid test index {idx}.")
            sys.exit(1)
    else:
        # Auto-select 4 representative test records representing different diagnostic configurations
        # We search for records that are positive for NORM, MI, STTC, CD, or HYP
        print("\n=== Auto-selecting 4 Representative Test Instances ===")
        for class_idx in range(len(LABELS)):
            matching_idxs = np.where(y_test[:, class_idx] == 1)[0]
            for idx in matching_idxs:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    break
            if len(selected_indices) >= 4:
                break
                
    reports_data = []
    for test_idx in selected_indices:
        record_id = int(test_record_ids[test_idx])
        print(f"\n------------------------------------------------------------")
        print(f"Processing Record ID: {record_id} (Test Index: {test_idx})")
        
        x_instance = X_test_raw[test_idx]
        true_label = y_test[test_idx]
        
        # Run explainability for this instance
        record_report = explain_instance_multilabel(
            explainer_wrapper, x_instance, true_label, record_id, thresholds, args.save_dir
        )
        reports_data.append(record_report)
        
    # Generate HTML report
    generate_html_report(reports_data, thresholds, args.save_dir)

def explain_instance_multilabel(explainer_wrapper, x_instance, true_label, record_id, thresholds, save_dir):
    # Get ground truth classes
    true_classes = [LABELS[i] for i in range(len(LABELS)) if true_label[i] == 1]
    print(f"Ground Truth Class(es): {true_classes if true_classes else 'None'}")
    
    # Compute prediction probabilities
    probs, shap_values_list = explainer_wrapper.explain_instance(x_instance)
    
    # Find predicted classes crossing thresholds
    pred_classes = []
    for i, class_name in enumerate(LABELS):
        if probs[i] >= thresholds[i]:
            pred_classes.append(class_name)
            
    # Fallback to class with highest probability if none cross threshold
    if not pred_classes:
        max_idx = np.argmax(probs)
        pred_classes = [LABELS[max_idx]]
        print(f"No class probability crossed threshold. Falling back to highest prob class: {pred_classes[0]}")
        
    print("\nPrediction Probabilities:")
    for i, class_name in enumerate(LABELS):
        status = " (PREDICTED)" if class_name in pred_classes else ""
        print(f"  - {class_name}: {probs[i]:.4f} [threshold: {thresholds[i]:.2f}]{status}")
        
    explanations = []
    
    # Generate a SHAP plot for every predicted class
    for class_name in pred_classes:
        class_idx = LABELS.index(class_name)
        print(f"\n---> Explaining Class: {class_name} (Prob: {probs[class_idx]:.4f})")
        
        # Get class SHAP
        if isinstance(shap_values_list, list):
            class_shap = shap_values_list[class_idx][0]
        else:
            class_shap = shap_values_list[0, :, class_idx]
            
        # Top contributors
        sorted_features_idx = np.argsort(class_shap)
        
        top_pos = []
        top_neg = []
        
        # Top 5 positive contributors
        for idx in reversed(sorted_features_idx):
            if class_shap[idx] > 1e-4:
                feat_val = x_instance[idx]
                val_str = f"{feat_val:.2f}" if not np.isnan(feat_val) else "NaN"
                top_pos.append((FEATURES[idx], float(class_shap[idx]), val_str))
                if len(top_pos) >= 5:
                    break
                    
        # Top 5 negative contributors
        for idx in sorted_features_idx:
            if class_shap[idx] < -1e-4:
                feat_val = x_instance[idx]
                val_str = f"{feat_val:.2f}" if not np.isnan(feat_val) else "NaN"
                top_neg.append((FEATURES[idx], float(class_shap[idx]), val_str))
                if len(top_neg) >= 5:
                    break
                    
        print("  * Top Positive Contributors:")
        for feat, val, val_str in top_pos:
            print(f"    - {feat}: +{val:.4f} (value: {val_str})")
        print("  * Top Negative Contributors:")
        for feat, val, val_str in top_neg:
            print(f"    - {feat}: {val:.4f} (value: {val_str})")
            
        # Plot and save
        img_filename = f"shap_explanation_record_{record_id}_{class_name}.png"
        plot_path = os.path.join(save_dir, img_filename)
        explainer_wrapper.plot_explanation(class_shap, x_instance, class_name, probs[class_idx], save_path=plot_path)
        
        explanations.append({
            "class_name": class_name,
            "prob": float(probs[class_idx]),
            "img_filename": img_filename,
            "top_pos": top_pos,
            "top_neg": top_neg
        })
        
    return {
        "record_id": record_id,
        "true_labels": true_classes,
        "probs": probs.tolist(),
        "explanations": explanations
    }

if __name__ == "__main__":
    main()
