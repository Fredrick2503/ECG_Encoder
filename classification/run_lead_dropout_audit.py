import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import brier_score_loss

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from data_management.dataset_factory import DatasetFactory
from temporal_encoder.encoder import ECGResNet1D as ECGResNet1D_SE
from morphology_encoder.encoder import ECGMorphologyEncoder
from morphology_encoder.conversion import ecg_to_spectrogram
from classification.classifier import ZFusedDataset, MLPClassifier
from classification.metrics import calculate_metrics
from config.constants import STANDARD_12_LEADS

# Settings
SEED = 42
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

torch.manual_seed(SEED)
np.random.seed(SEED)

def main():
    print("Loading test dataset splits...")
    _, _, test_ds, loader = DatasetFactory.create_datasets(
        dataset_type="ptbxl", download=False, resolution="lr"
    )
    
    data_file = project_root / "data" / "Z_fused_2k.npz"
    data = np.load(data_file)
    test_record_ids = data["test_record_id"]
    test_labels = data["test_labels"]
    
    record_to_index = {int(rid): idx for idx, rid in enumerate(test_ds.record_ids)}
    test_idx_mapping = [record_to_index[int(rid)] for rid in test_record_ids]
    n_test = len(test_idx_mapping)
    
    # Load Models
    print("Loading models...")
    temp_model = ECGResNet1D_SE(num_classes=5, use_se=True).to(device)
    temp_model.load_state_dict(torch.load(project_root / "models/C5_full_dataset.pt", map_location=device))
    temp_model.eval()
    
    morph_model = ECGMorphologyEncoder(input_channels=12, num_classes=5).to(device)
    morph_model.load_state_dict(torch.load(project_root / "models/morphology_encoder_v1.pt", map_location=device))
    morph_model.eval()
    
    from biomarkers.models import AttentionMLPAutoencoder
    bio_model = AttentionMLPAutoencoder(input_dim=50, latent_dim=32).to(device)
    bio_model.load_state_dict(torch.load(project_root / "biomarkers/attention_mlp_best.pt", map_location=device))
    bio_model.eval()
    
    classifier_model = MLPClassifier(input_dim=1056, hidden_dim=256, num_classes=5).to(device)
    classifier_model.load_state_dict(torch.load(project_root / "models/classification_mlp.pt", map_location=device))
    classifier_model.eval()
    
    thrs = np.load(project_root / "models/classification_mlp_thresholds.npy")
    
    # Load Biomarkers
    FEATURES = [
        "RR_Mean", "QRS_Duration", "PR_Interval", "QT_Interval", "QTc_Bazett",
        "ST_Duration", "P_wave_Duration", "R_Amplitude", "P_Amplitude", "T_Amplitude",
        "ST_Deviation", "Q_Amplitude", "R_S_Ratio", "QRS_Energy", "SDNN",
        "RMSSD", "pNN50", "pNN20", "SDRR_RMSSD_Ratio", "HRV_Triangular_Index",
        "LF_Power", "HF_Power", "LF_HF_Ratio", "Total_Power", "Sample_Entropy"
    ]
    biomarkers_csv = project_root / "biomarkers" / "ecg_biomarkers_full.csv"
    df_bio = pd.read_csv(biomarkers_csv)
    df_bio.set_index("record_id", inplace=True)
    with open(project_root / "biomarkers" / "imputer.pkl", "rb") as f:
        imputer = pickle.load(f)
    with open(project_root / "biomarkers" / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    df_bio_features = df_bio[FEATURES]
    X_imputed = imputer.transform(df_bio_features)
    X_scaled = scaler.transform(X_imputed)
    M = (~df_bio_features.isna()).astype(np.float32).values
    X_joint = np.concatenate([X_scaled, M], axis=1)
    bio_record_ids = df_bio.index.values.tolist()
    record_to_bio_idx = {int(rid): idx for idx, rid in enumerate(bio_record_ids)}
    
    def extract_fused(mask_leads_indices=None, keep_leads_indices=None):
        zt_list, zm_list, zb_list = [], [], []
        with torch.no_grad():
            for i in range(n_test):
                orig_idx = test_idx_mapping[i]
                rid = int(test_record_ids[i])
                signal, _ = test_ds[orig_idx]
                signal_np = signal.numpy() if hasattr(signal, "numpy") else signal
                
                signal_np_masked = signal_np.copy()
                if mask_leads_indices is not None:
                    for idx in mask_leads_indices:
                        signal_np_masked[idx, :] = 0.0
                elif keep_leads_indices is not None:
                    # Mask everything else
                    for idx in range(12):
                        if idx not in keep_leads_indices:
                            signal_np_masked[idx, :] = 0.0
                            
                bx = torch.tensor(signal_np_masked, dtype=torch.float32).unsqueeze(0).to(device)
                
                zt = temp_model.get_representation(bx)
                zt_list.append(zt.cpu().numpy())
                
                spec = ecg_to_spectrogram(bx)
                zm = morph_model.get_representation(spec)
                zm_list.append(zm.cpu().numpy())
                
                if rid in record_to_bio_idx:
                    bio_idx = record_to_bio_idx[rid]
                    bio_feat = X_joint[bio_idx]
                else:
                    bio_feat = np.zeros(50, dtype=np.float32)
                bio_tensor = torch.tensor(bio_feat, dtype=torch.float32).unsqueeze(0).to(device)
                zb = bio_model.encode(bio_tensor)
                zb_list.append(zb.cpu().numpy())
                
        return np.concatenate([np.concatenate(zt_list), np.concatenate(zm_list), np.concatenate(zb_list)], axis=1)

    def evaluate_mask(mask_name, mask_idx=None, keep_idx=None):
        z_feat = extract_fused(mask_leads_indices=mask_idx, keep_leads_indices=keep_idx)
        test_loader = DataLoader(ZFusedDataset(z_feat, test_labels), batch_size=BATCH_SIZE, shuffle=False)
        all_probs = []
        with torch.no_grad():
            for batch in test_loader:
                bz = batch["z"].to(device)
                probs = torch.sigmoid(classifier_model(bz))
                all_probs.append(probs.cpu().numpy())
        all_probs = np.concatenate(all_probs, axis=0)
        
        metrics = calculate_metrics(test_labels, all_probs, thrs)
        briers = [brier_score_loss(test_labels[:, c], all_probs[:, c]) for c in range(5)]
        
        return {
            "Configuration": mask_name,
            "Macro F1": metrics["macro_f1"],
            "Macro AUC": metrics["macro_auc"],
            "Subset Acc": metrics["subset_acc"],
            "NORM F1": metrics["per_class_f1"][0],
            "MI F1": metrics["per_class_f1"][1],
            "STTC F1": metrics["per_class_f1"][2],
            "CD F1": metrics["per_class_f1"][3],
            "HYP F1": metrics["per_class_f1"][4],
            "Brier Score": np.mean(briers)
        }

    print("Evaluating clean baseline...")
    results = [evaluate_mask("Clean Baseline")]
    
    # 1. Single Lead Dropout (Mask each lead individually)
    for idx, lead in enumerate(STANDARD_12_LEADS):
        print(f"Auditing Lead Dropout: {lead}...")
        results.append(evaluate_mask(f"Drop {lead}", mask_idx=[idx]))
        
    # 2. Lead Group Dropout
    groups = {
        "Drop I-III (Limb Leads)": [0, 1, 2],
        "Drop aVR/aVL/aVF (Augmented)": [3, 4, 5],
        "Drop V1-V6 (Chest Leads)": [6, 7, 8, 9, 10, 11]
    }
    for g_name, g_idx in groups.items():
        print(f"Auditing Group Dropout: {g_name}...")
        results.append(evaluate_mask(g_name, mask_idx=g_idx))
        
    # 3. Solo Lead/Group (Keep only that lead/group)
    for idx, lead in enumerate(STANDARD_12_LEADS):
        print(f"Auditing Solo Lead: Only {lead}...")
        results.append(evaluate_mask(f"Solo {lead}", keep_idx=[idx]))
        
    solo_groups = {
        "Solo I-III (Limb Leads)": [0, 1, 2],
        "Solo aVR/aVL/aVF (Augmented)": [3, 4, 5],
        "Solo V1-V6 (Chest Leads)": [6, 7, 8, 9, 10, 11]
    }
    for g_name, g_idx in solo_groups.items():
        print(f"Auditing Solo Group: {g_name}...")
        results.append(evaluate_mask(g_name, keep_idx=g_idx))
        
    df_results = pd.DataFrame(results)
    os.makedirs(project_root / "outputs/reports", exist_ok=True)
    report_csv = project_root / "outputs/reports/lead_dropout_audit_results.csv"
    df_results.to_csv(report_csv, index=False)
    print(f"Saved raw CSV results to {report_csv}")
    
    # Generate lead dependency matrix heatmap or table format
    # Let's format the table nicely for the markdown report
    report_path = project_root / "outputs/reports/lead_dropout_audit.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Lead-Dropout Audit & Dependency Matrix Report\n\n")
        f.write("This report evaluates the clinical and classification dependency of Model B on specific ECG leads and lead groups.\n\n")
        
        f.write("## 1. Single Lead Dropout Analysis (Removing a single lead)\n\n")
        df_single_drop = df_results[df_results["Configuration"].str.startswith("Drop ") & ~df_results["Configuration"].str.contains("-|/")].copy()
        df_single_drop["F1 Delta"] = df_single_drop["Macro F1"] - results[0]["Macro F1"]
        f.write(df_single_drop[["Configuration", "Macro F1", "F1 Delta", "NORM F1", "MI F1", "STTC F1", "CD F1", "HYP F1"]].to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 2. Lead Group Dropout Analysis (Removing a subset of leads)\n\n")
        df_group_drop = df_results[df_results["Configuration"].str.startswith("Drop ") & df_results["Configuration"].str.contains("-|/")].copy()
        df_group_drop["F1 Delta"] = df_group_drop["Macro F1"] - results[0]["Macro F1"]
        f.write(df_group_drop[["Configuration", "Macro F1", "F1 Delta", "NORM F1", "MI F1", "STTC F1", "CD F1", "HYP F1"]].to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 3. Solo Lead/Group Analysis (Keeping ONLY that lead/group)\n\n")
        df_solo = df_results[df_results["Configuration"].str.startswith("Solo")].copy()
        f.write(df_solo[["Configuration", "Macro F1", "NORM F1", "MI F1", "STTC F1", "CD F1", "HYP F1"]].to_markdown(index=False))
        f.write("\n\n")
        
    print(f"Saved report to {report_path}")

if __name__ == "__main__":
    main()
