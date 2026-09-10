import os
import sys
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss

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
from explainability.unified_explainer import EndToEndUnifiedModel, UnifiedExplainer

# Settings
SEED = 42
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

# Noise overlay functions
def apply_baseline_wander(x, sr=100, f=0.5, A=0.3):
    t = np.arange(x.shape[-1]) / sr
    wander = A * np.sin(2 * np.pi * f * t)
    return x + wander

def apply_powerline_interference(x, sr=100, f=50, A=0.15):
    t = np.arange(x.shape[-1]) / sr
    noise = A * np.sin(2 * np.pi * f * t)
    return x + noise

def apply_high_frequency_noise(x, sigma=0.05):
    noise = np.random.normal(0, sigma, x.shape)
    return x + noise

def apply_single_lead_mask(x, lead_idx=1):
    x_masked = x.copy()
    x_masked[lead_idx, :] = 0.0
    return x_masked

def apply_multiple_leads_mask(x, lead_indices=range(6, 12)):
    x_masked = x.copy()
    for idx in lead_indices:
        x_masked[idx, :] = 0.0
    return x_masked

def main():
    print("Loading test dataset splits...")
    _, _, test_ds, loader = DatasetFactory.create_datasets(
        dataset_type="ptbxl", download=False, resolution="lr"
    )
    
    data_file = project_root / "data" / "Z_fused_2k.npz"
    data = np.load(data_file)
    test_record_ids = data["test_record_id"]
    test_labels = data["test_labels"]
    
    # Map test record IDs to their indices in the test_ds
    record_to_index = {int(rid): idx for idx, rid in enumerate(test_ds.record_ids)}
    test_idx_mapping = [record_to_index[int(rid)] for rid in test_record_ids]
    n_test = len(test_idx_mapping)
    
    # Load Models
    print("Loading frozen sub-encoders...")
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
    print(f"Loaded threshold arrays: {thrs}")

    # Load Biomarker Alignments
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
    X_joint = np.concatenate([X_scaled, M], axis=1) # 50 dimensions
    bio_record_ids = df_bio.index.values.tolist()
    record_to_bio_idx = {int(rid): idx for idx, rid in enumerate(bio_record_ids)}
    
    # Function to extract Z_fused on custom signals
    def extract_fused_representations(perturbation_fn=None, mask_temporal=False, mask_morphology=False, mask_biomarker=False):
        zt_list, zm_list, zb_list = [], [], []
        
        with torch.no_grad():
            for i in range(n_test):
                orig_idx = test_idx_mapping[i]
                rid = int(test_record_ids[i])
                signal, _ = test_ds[orig_idx]
                signal_np = signal.numpy() if hasattr(signal, "numpy") else signal
                
                # Apply custom perturbation
                if perturbation_fn is not None:
                    signal_np = perturbation_fn(signal_np)
                    
                bx = torch.tensor(signal_np, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Get temporal representation
                if mask_temporal:
                    zt = torch.zeros((1, 512), device=device)
                else:
                    zt = temp_model.get_representation(bx)
                zt_list.append(zt.cpu().numpy())
                
                # Get morphology representation
                if mask_morphology:
                    zm = torch.zeros((1, 512), device=device)
                else:
                    spec = ecg_to_spectrogram(bx)
                    zm = morph_model.get_representation(spec)
                zm_list.append(zm.cpu().numpy())

                # Get biomarker representation
                if mask_biomarker:
                    zb = torch.zeros((1, 32), device=device)
                else:
                    if rid in record_to_bio_idx:
                        bio_idx = record_to_bio_idx[rid]
                        bio_feat = X_joint[bio_idx]
                    else:
                        bio_feat = np.zeros(50, dtype=np.float32)
                    bio_tensor = torch.tensor(bio_feat, dtype=torch.float32).unsqueeze(0).to(device)
                    zb = bio_model.encode(bio_tensor)
                zb_list.append(zb.cpu().numpy())
                
        zt_arr = np.concatenate(zt_list, axis=0)
        zm_arr = np.concatenate(zm_list, axis=0)
        zb_arr = np.concatenate(zb_list, axis=0)
        z_fused = np.concatenate([zt_arr, zm_arr, zb_arr], axis=1) # 1056-D
        return z_fused

    # 1. Clean test baseline
    print("\nEvaluating Clean Test baseline...")
    z_clean = extract_fused_representations(perturbation_fn=None)
    
    # ─── RUN BENCHMARKS ON ALL PERTURBATIONS ──────────────────────────────────
    perturbations = {
        "Clean Baseline": None,
        "Baseline Wander": lambda x: apply_baseline_wander(x, A=0.3),
        "Powerline Interference": lambda x: apply_powerline_interference(x, A=0.15),
        "High-Frequency Noise": lambda x: apply_high_frequency_noise(x, sigma=0.05),
        "Single-Lead Masking": lambda x: apply_single_lead_mask(x, lead_idx=1),
        "Chest-Leads Masking": lambda x: apply_multiple_leads_mask(x, lead_indices=range(6, 12)),
        "Combined Perturbation": lambda x: apply_multiple_leads_mask(apply_high_frequency_noise(apply_baseline_wander(x, A=0.2), sigma=0.03), range(6, 12))
    }
    
    results = []
    
    for name, fn in perturbations.items():
        print(f"Evaluating: {name}...")
        z_feat = extract_fused_representations(perturbation_fn=fn)
        
        # Evaluate using MLP classifier
        test_loader = DataLoader(ZFusedDataset(z_feat, test_labels), batch_size=BATCH_SIZE, shuffle=False)
        all_probs = []
        with torch.no_grad():
            for batch in test_loader:
                bz = batch["z"].to(device)
                probs = torch.sigmoid(classifier_model(bz))
                all_probs.append(probs.cpu().numpy())
        all_probs = np.concatenate(all_probs, axis=0)
        
        # Metrics
        metrics = calculate_metrics(test_labels, all_probs, thrs)
        briers = [brier_score_loss(test_labels[:, c], all_probs[:, c]) for c in range(5)]
        mean_brier = np.mean(briers)
        
        res = {
            "Perturbation": name,
            "Macro F1": metrics["macro_f1"],
            "Macro AUC": metrics["macro_auc"],
            "Subset Accuracy": metrics["subset_acc"],
            "Macro ECE": metrics["macro_ece"],
            "Brier Score": mean_brier
        }
        results.append(res)
        
    df_perf = pd.DataFrame(results)
    
    # Calculate performance deltas from clean baseline
    baseline_f1 = df_perf.loc[df_perf["Perturbation"] == "Clean Baseline", "Macro F1"].values[0]
    df_perf["F1 Delta"] = df_perf["Macro F1"] - baseline_f1
    
    # ─── MODALITY DEGRADATION & RESILIENCE ────────────────────────────────────
    print("\n--- Modality Degradation & Resilience analysis ---")
    
    # 1. Temporal Zeroed
    z_temp_zeroed = z_clean.copy()
    z_temp_zeroed[:, 0:512] = 0.0
    
    # 2. Morphology Zeroed
    z_morph_zeroed = z_clean.copy()
    z_morph_zeroed[:, 512:1024] = 0.0

    # 3. Biomarker Zeroed
    z_bio_zeroed = z_clean.copy()
    z_bio_zeroed[:, 1024:1056] = 0.0
    
    degradations = {
        "Temporal Zeroed (Morphology + Biomarker)": z_temp_zeroed,
        "Morphology Zeroed (Temporal + Biomarker)": z_morph_zeroed,
        "Biomarker Zeroed (Temporal + Morphology)": z_bio_zeroed
    }
    
    deg_results = []
    for name, z_feat in degradations.items():
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
        mean_brier = np.mean(briers)
        
        res = {
            "Perturbation": name,
            "Macro F1": metrics["macro_f1"],
            "Macro AUC": metrics["macro_auc"],
            "Subset Accuracy": metrics["subset_acc"],
            "Macro ECE": metrics["macro_ece"],
            "Brier Score": mean_brier,
            "F1 Delta": metrics["macro_f1"] - baseline_f1
        }
        deg_results.append(res)
        
    df_deg = pd.DataFrame(deg_results)
    df_all = pd.concat([df_perf, df_deg], ignore_index=True)
    
    print("\nFinal Robustness Comparison Table:")
    print(df_all.to_string(index=False))
    
    # ─── XAI STABILITY ANALYSIS ───────────────────────────────────────────────
    print("\n--- XAI Stability Evaluation under Noise ---")
    
    # Wrapper mapping signal_1d to Model B logits
    class WrapperModelB(nn.Module):
        def __init__(self, temp_model, morph_model, bio_model, classifier_model, bio_tensor):
            super().__init__()
            self.temp_model = temp_model
            self.morph_model = morph_model
            self.bio_model = bio_model
            self.classifier_model = classifier_model
            self.bio_tensor = bio_tensor
            
        def forward(self, signal_1d):
            zt = self.temp_model.get_representation(signal_1d)
            spec = ecg_to_spectrogram(signal_1d)
            zm = self.morph_model.get_representation(spec)
            # Match batch size
            bio_t = self.bio_tensor.to(signal_1d.device)
            if bio_t.size(0) != zt.size(0):
                bio_t = bio_t.expand(zt.size(0), -1)
            zb = self.bio_model.encode(bio_t)
            z_fused_b = torch.cat([zt, zm, zb], dim=1)
            return self.classifier_model(z_fused_b)
            
    from captum.attr import IntegratedGradients
    
    xai_samples = 10
    cosine_similarities = []
    
    for i in range(xai_samples):
        orig_idx = test_idx_mapping[i]
        rid = int(test_record_ids[i])
        signal, _ = test_ds[orig_idx]
        sig_tensor = signal.unsqueeze(0).to(device) # (1, 12, 1000)
        
        # Load bio tensor
        if rid in record_to_bio_idx:
            bio_feat = X_joint[record_to_bio_idx[rid]]
        else:
            bio_feat = np.zeros(50, dtype=np.float32)
        bio_tensor = torch.tensor(bio_feat, dtype=torch.float32).unsqueeze(0).to(device)
        
        wrapper_b = WrapperModelB(temp_model, morph_model, bio_model, classifier_model, bio_tensor).to(device)
        wrapper_b.eval()
        ig_b = IntegratedGradients(wrapper_b)
        
        # Predict target class
        with torch.no_grad():
            logits = wrapper_b(sig_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        target_class = int(np.argmax(probs))
        
        # Clean attribution
        sig_tensor.requires_grad = True
        baseline_sig = torch.zeros_like(sig_tensor).to(device)
        attr_clean = ig_b.attribute(sig_tensor, baselines=baseline_sig, target=target_class, n_steps=15).detach().cpu().numpy()[0]
        
        # Perturbed attribution (with High-Frequency noise)
        perturbed_sig_np = apply_high_frequency_noise(signal.numpy(), sigma=0.03)
        perturbed_sig_tensor = torch.tensor(perturbed_sig_np, dtype=torch.float32).unsqueeze(0).to(device)
        perturbed_sig_tensor.requires_grad = True
        
        baseline_sig_perturbed = torch.zeros_like(perturbed_sig_tensor).to(device)
        attr_perturbed = ig_b.attribute(perturbed_sig_tensor, baselines=baseline_sig_perturbed, target=target_class, n_steps=15).detach().cpu().numpy()[0]
        
        # Compute cosine similarity of absolute attributions
        v_clean = np.abs(attr_clean).flatten()
        v_perturbed = np.abs(attr_perturbed).flatten()
        
        cosine_sim = np.dot(v_clean, v_perturbed) / (np.linalg.norm(v_clean) * np.linalg.norm(v_perturbed) + 1e-8)
        cosine_similarities.append(float(cosine_sim))
        
    mean_cosine_stability = np.mean(cosine_similarities)
    print(f"Mean Cosine XAI Stability under Noise: {mean_cosine_stability:.4f}")
    
    # ─── SAVE VISUALIZATION PLOT ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    noise_names = df_perf["Perturbation"].tolist()
    f1_scores = df_perf["Macro F1"].tolist()
    
    ax.barh(noise_names, f1_scores, color='#ff0d57', height=0.5)
    ax.axvline(baseline_f1, color='#1e88e5', linestyle='--', label=f'Baseline F1 ({baseline_f1:.4f})')
    
    ax.set_title('Primary Model B (T+M+B) Robustness under Signal Perturbations')
    ax.set_xlabel('Macro F1 Score')
    ax.set_xlim(0, 0.8)
    ax.legend()
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    fig_dir = project_root / "outputs/figures"
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = fig_dir / "robustness_degradation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # ─── SAVE REPORTS ─────────────────────────────────────────────────────────
    report_dir = project_root / "outputs/reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = report_dir / "robustness_validation_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 6: Robustness & Failure-Mode Validation Report\n\n")
        f.write("This benchmark study evaluates the noise resilience, lead-masking sensitivity, modality branch zeroing impact, and XAI stability of the primary Model B (T+M+B) classifier.\n\n")
        
        f.write("## 1. Perturbation Performance Summary\n\n")
        f.write(df_all.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 2. Modality Compensation Analysis\n\n")
        f.write(f"- **Temporal Branch Zeroed**: Macro F1 drops by **`{-df_all.loc[df_all['Perturbation'] == 'Temporal Zeroed (Morphology + Biomarker)', 'F1 Delta'].values[0]:.4f}`**.\n")
        f.write(f"- **Morphology Branch Zeroed**: Macro F1 drops by **`{-df_all.loc[df_all['Perturbation'] == 'Morphology Zeroed (Temporal + Biomarker)', 'F1 Delta'].values[0]:.4f}`**.\n")
        f.write(f"- **Biomarker Branch Zeroed**: Macro F1 drops by **`{-df_all.loc[df_all['Perturbation'] == 'Biomarker Zeroed (Temporal + Morphology)', 'F1 Delta'].values[0]:.4f}`**.\n\n")
        
        f.write("## 3. Explainable AI Stability under Perturbation\n\n")
        f.write(f"- **Mean Cosine XAI Stability score**: **`{mean_cosine_stability:.4f}`** (calculated between absolute attributions of clean and high-frequency noise signals across 10 samples).\n\n")
        
        f.write("## 4. Visualized Degradation Curves\n")
        f.write("The horizontal bar plot comparing Macro F1 scores under all perturbations is saved to `outputs/figures/robustness_degradation.png`.\n")
        
    print(f"Saved report to {report_path}")

if __name__ == "__main__":
    main()
