import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
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

# Settings
SEED = 42
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

torch.manual_seed(SEED)
np.random.seed(SEED)

# Corruption Functions
def apply_baseline_wander(x, A=0.3, sr=100, f=0.5):
    t = np.arange(x.shape[-1]) / sr
    wander = A * np.sin(2 * np.pi * f * t)
    return x + wander

def apply_powerline_interference(x, A=0.15, sr=100, f=50):
    t = np.arange(x.shape[-1]) / sr
    noise = A * np.sin(2 * np.pi * f * t)
    return x + noise

def apply_high_frequency_noise(x, sigma=0.05):
    noise = np.random.normal(0, sigma, x.shape)
    return x + noise

def apply_amplitude_scaling(x, scale=1.0):
    return x * scale

def apply_baseline_offset(x, offset=0.3):
    return x + offset

def apply_emg_noise(x, sigma=0.05):
    # EMG noise modeled as high-frequency noise with time-varying envelope (bursts)
    n_samples = x.shape[-1]
    t = np.arange(n_samples)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * t / 100) # slow modulation
    noise = np.random.normal(0, sigma, x.shape) * envelope
    return x + noise

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
    print("Loading frozen encoders and classifier...")
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
    
    def extract_fused(perturbation_fn):
        zt_list, zm_list, zb_list = [], [], []
        with torch.no_grad():
            for i in range(n_test):
                orig_idx = test_idx_mapping[i]
                rid = int(test_record_ids[i])
                signal, _ = test_ds[orig_idx]
                signal_np = signal.numpy() if hasattr(signal, "numpy") else signal
                
                if perturbation_fn is not None:
                    signal_np = perturbation_fn(signal_np)
                    
                bx = torch.tensor(signal_np, dtype=torch.float32).unsqueeze(0).to(device)
                
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

    # Clean test benchmark first
    print("Running Clean baseline...")
    z_clean = extract_fused(None)
    test_loader = DataLoader(ZFusedDataset(z_clean, test_labels), batch_size=BATCH_SIZE, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            bz = batch["z"].to(device)
            probs = torch.sigmoid(classifier_model(bz))
            all_probs.append(probs.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    baseline_metrics = calculate_metrics(test_labels, all_probs, thrs)
    baseline_f1 = baseline_metrics["macro_f1"]
    baseline_auc = baseline_metrics["macro_auc"]
    baseline_ece = baseline_metrics["macro_ece"]
    baseline_brier = np.mean([brier_score_loss(test_labels[:, c], all_probs[:, c]) for c in range(5)])

    # Setup Severities
    severities = [1, 2, 3, 4, 5]
    corruptions = {
        "Baseline Wander": {
            1: lambda x: apply_baseline_wander(x, A=0.05),
            2: lambda x: apply_baseline_wander(x, A=0.15),
            3: lambda x: apply_baseline_wander(x, A=0.30),
            4: lambda x: apply_baseline_wander(x, A=0.60),
            5: lambda x: apply_baseline_wander(x, A=1.20)
        },
        "Powerline Interference": {
            1: lambda x: apply_powerline_interference(x, A=0.02),
            2: lambda x: apply_powerline_interference(x, A=0.06),
            3: lambda x: apply_powerline_interference(x, A=0.15),
            4: lambda x: apply_powerline_interference(x, A=0.30),
            5: lambda x: apply_powerline_interference(x, A=0.60)
        },
        "High-Frequency Noise": {
            1: lambda x: apply_high_frequency_noise(x, sigma=0.01),
            2: lambda x: apply_high_frequency_noise(x, sigma=0.03),
            3: lambda x: apply_high_frequency_noise(x, sigma=0.07),
            4: lambda x: apply_high_frequency_noise(x, sigma=0.15),
            5: lambda x: apply_high_frequency_noise(x, sigma=0.30)
        },
        "Amplitude Scaling": {
            1: lambda x: apply_amplitude_scaling(x, scale=0.9),
            2: lambda x: apply_amplitude_scaling(x, scale=0.7),
            3: lambda x: apply_amplitude_scaling(x, scale=0.5),
            4: lambda x: apply_amplitude_scaling(x, scale=0.3),
            5: lambda x: apply_amplitude_scaling(x, scale=0.1)
        },
        "Baseline Offset": {
            1: lambda x: apply_baseline_offset(x, offset=0.1),
            2: lambda x: apply_baseline_offset(x, offset=0.3),
            3: lambda x: apply_baseline_offset(x, offset=0.7),
            4: lambda x: apply_baseline_offset(x, offset=1.5),
            5: lambda x: apply_baseline_offset(x, offset=3.0)
        },
        "EMG Noise": {
            1: lambda x: apply_emg_noise(x, sigma=0.01),
            2: lambda x: apply_emg_noise(x, sigma=0.03),
            3: lambda x: apply_emg_noise(x, sigma=0.07),
            4: lambda x: apply_emg_noise(x, sigma=0.15),
            5: lambda x: apply_emg_noise(x, sigma=0.30)
        }
    }
    
    results = []
    
    # Store clean baseline in results for level 0
    for corr_name in corruptions.keys():
        results.append({
            "Corruption": corr_name,
            "Severity": 0,
            "Macro F1": baseline_f1,
            "Macro AUC": baseline_auc,
            "Macro ECE": baseline_ece,
            "Brier Score": baseline_brier
        })

    for corr_name, levels in corruptions.items():
        for lvl, fn in levels.items():
            print(f"Running {corr_name} Severity {lvl}...")
            z_corr = extract_fused(fn)
            test_loader = DataLoader(ZFusedDataset(z_corr, test_labels), batch_size=BATCH_SIZE, shuffle=False)
            all_probs = []
            with torch.no_grad():
                for batch in test_loader:
                    bz = batch["z"].to(device)
                    probs = torch.sigmoid(classifier_model(bz))
                    all_probs.append(probs.cpu().numpy())
            all_probs = np.concatenate(all_probs, axis=0)
            metrics = calculate_metrics(test_labels, all_probs, thrs)
            brier = np.mean([brier_score_loss(test_labels[:, c], all_probs[:, c]) for c in range(5)])
            
            results.append({
                "Corruption": corr_name,
                "Severity": lvl,
                "Macro F1": metrics["macro_f1"],
                "Macro AUC": metrics["macro_auc"],
                "Macro ECE": metrics["macro_ece"],
                "Brier Score": brier
            })
            
    df_results = pd.DataFrame(results)
    os.makedirs(project_root / "outputs/reports", exist_ok=True)
    df_results.to_csv(project_root / "outputs/reports/severity_robustness_results.csv", index=False)
    
    # Generate Plots
    print("Generating severity robustness curves...")
    os.makedirs(project_root / "outputs/figures", exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics_to_plot = ["Macro F1", "Macro AUC", "Macro ECE", "Brier Score"]
    axes_flat = axes.flatten()
    
    colors = {
        "Baseline Wander": "#1f77b4",
        "Powerline Interference": "#ff7f0e",
        "High-Frequency Noise": "#2ca02c",
        "Amplitude Scaling": "#d62728",
        "Baseline Offset": "#9467bd",
        "EMG Noise": "#8c564b"
    }
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes_flat[idx]
        for corr_name in corruptions.keys():
            df_sub = df_results[df_results["Corruption"] == corr_name].sort_values("Severity")
            ax.plot(df_sub["Severity"], df_sub[metric], marker='o', label=corr_name, color=colors[corr_name])
            
        ax.set_title(f"{metric} vs. Noise Severity")
        ax.set_xlabel("Severity Level (0 = Clean)")
        ax.set_ylabel(metric)
        ax.grid(True, linestyle="--", alpha=0.6)
        if idx == 0:
            ax.legend()
            
    plt.tight_layout()
    plt.savefig(project_root / "outputs/figures/severity_robustness_curves.png", dpi=150)
    plt.close()
    
    # Generate Failure Ranking
    # Ranking is based on the drop in Macro F1 from level 0 to level 5 (or level 3 for a milder view)
    ranking_data = []
    for corr_name in corruptions.keys():
        f1_0 = df_results[(df_results["Corruption"] == corr_name) & (df_results["Severity"] == 0)]["Macro F1"].values[0]
        f1_5 = df_results[(df_results["Corruption"] == corr_name) & (df_results["Severity"] == 5)]["Macro F1"].values[0]
        f1_drop = f1_0 - f1_5
        ranking_data.append({
            "Corruption": corr_name,
            "F1 at Level 0 (Clean)": f1_0,
            "F1 at Level 5 (Severe)": f1_5,
            "Macro F1 Drop": f1_drop
        })
    df_ranking = pd.DataFrame(ranking_data).sort_values("Macro F1 Drop", ascending=False)
    
    # Write report
    report_path = project_root / "outputs/reports/severity_robustness_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Corruption Severity Benchmark Report\n\n")
        f.write("This report details the model robustness across 5 severity levels of signal corruption.\n\n")
        
        f.write("## 1. Failure-Mode Impact Ranking (F1 Drop from Clean to Level 5)\n\n")
        f.write(df_ranking.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 2. Detailed Performance Table\n\n")
        f.write(df_results.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 3. Visualized Curves\n")
        f.write("The plots showing F1, AUC, ECE, and Brier score against severity are saved to `outputs/figures/severity_robustness_curves.png`.\n")
        
    print(f"Saved report to {report_path}")

if __name__ == "__main__":
    main()
