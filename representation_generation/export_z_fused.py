import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Subset

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from data_management.dataset_factory import DatasetFactory
from temporal_encoder.encoder import ECGResNet1D as ECGResNet1D_SE
from morphology_encoder.encoder import ECGMorphologyEncoder
from morphology_encoder.conversion import ecg_to_spectrogram
from biomarkers.models import AttentionMLPAutoencoder

# Settings
BATCH_SIZE = 64
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def main():
    print(f"Using device: {device}")
    
    # 1. Load data
    print("Loading PTB-XL datasets...")
    train_ds, val_ds, test_ds, loader = DatasetFactory.create_datasets(
        dataset_type="ptbxl", download=False, resolution="lr"
    )
    
    # Subset to the 2K benchmark split
    train_n_ds = len(train_ds)
    val_n_ds = len(val_ds)
    test_n_ds = len(test_ds)
    
    n_train = min(2000, train_n_ds)
    n_val = min(max(150, int(n_train * 0.15)), val_n_ds)
    n_test = min(max(150, int(n_train * 0.15)), test_n_ds)
    
    g = torch.Generator().manual_seed(SEED)
    train_idx = torch.randperm(train_n_ds, generator=g)[:n_train].tolist()
    val_idx = torch.randperm(val_n_ds, generator=g)[:n_val].tolist()
    test_idx = torch.randperm(test_n_ds, generator=g)[:n_test].tolist()
    
    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(val_ds, val_idx)
    test_subset = Subset(test_ds, test_idx)
    
    # Load raw biomarker CSV to align features
    biomarkers_csv = project_root / "biomarkers" / "ecg_biomarkers_full.csv"
    print(f"Loading biomarkers from {biomarkers_csv}...")
    df_bio = pd.read_csv(biomarkers_csv)
    df_bio.set_index("record_id", inplace=True)
    
    # Load imputer and scaler
    with open(project_root / "biomarkers" / "imputer.pkl", "rb") as f:
        imputer = pickle.load(f)
    with open(project_root / "biomarkers" / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        
    FEATURES = [
        "RR_Mean", "QRS_Duration", "PR_Interval", "QT_Interval", "QTc_Bazett",
        "ST_Duration", "P_wave_Duration", "R_Amplitude", "P_Amplitude", "T_Amplitude",
        "ST_Deviation", "Q_Amplitude", "R_S_Ratio", "QRS_Energy", "SDNN",
        "RMSSD", "pNN50", "pNN20", "SDRR_RMSSD_Ratio", "HRV_Triangular_Index",
        "LF_Power", "HF_Power", "LF_HF_Ratio", "Total_Power", "Sample_Entropy"
    ]

    # Pre-impute and scale the biomarker dataframe
    df_bio_features = df_bio[FEATURES]
    X_imputed = imputer.transform(df_bio_features)
    X_scaled = scaler.transform(X_imputed)
    M = (~df_bio_features.isna()).astype(np.float32).values
    X_joint = np.concatenate([X_scaled, M], axis=1) # 50 dimensions
    
    # Map index to feature index for fast lookup
    bio_record_ids = df_bio.index.values.tolist()
    record_to_bio_idx = {int(rid): idx for idx, rid in enumerate(bio_record_ids)}

    # 2. Load models
    print("Loading models...")
    temp_model = ECGResNet1D_SE(num_classes=5, use_se=True).to(device)
    temp_model.load_state_dict(torch.load(project_root / "models/C5_full_dataset.pt", map_location=device))
    temp_model.eval()
    
    morph_model = ECGMorphologyEncoder(input_channels=12, num_classes=5).to(device)
    morph_model.load_state_dict(torch.load(project_root / "models/morphology_encoder_v1.pt", map_location=device))
    morph_model.eval()
    
    bio_model = AttentionMLPAutoencoder(input_dim=50, latent_dim=32).to(device)
    bio_model.load_state_dict(torch.load(project_root / "biomarkers/attention_mlp_best.pt", map_location=device))
    bio_model.eval()

    # Freeze all models
    for p in temp_model.parameters():
        p.requires_grad_(False)
    for p in morph_model.parameters():
        p.requires_grad_(False)
    for p in bio_model.parameters():
        p.requires_grad_(False)

    splits = {
        "train": train_subset,
        "val": val_subset,
        "test": test_subset
    }
    
    exported_data = {}
    
    for split_name, ds in splits.items():
        print(f"Extracting representations for split: {split_name} (N={len(ds)})...")
        loader_split = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        
        zt_list, zm_list, zb_list = [], [], []
        labels_list = []
        record_ids_list = []
        patient_ids_list = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader_split):
                bx, by = batch
                
                # Retrieve record IDs from the dataset subset slice
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(ds))
                # For Subset class, ds.indices contains the original indices mapping
                subset_indices = ds.indices[start_idx:end_idx]
                batch_records = [ds.dataset.record_ids[i] for i in subset_indices]
                
                # Fetch patient IDs and align biomarkers
                batch_patients = []
                batch_bio_feats = []
                
                for rid in batch_records:
                    rec = ds.dataset.loader.load_record(int(rid))
                    batch_patients.append(rec.patient_id if rec.patient_id is not None else -1)
                    
                    r_id_int = int(rid)
                    if r_id_int in record_to_bio_idx:
                        bio_idx = record_to_bio_idx[r_id_int]
                        batch_bio_feats.append(X_joint[bio_idx])
                    else:
                        batch_bio_feats.append(np.zeros(48, dtype=np.float32))
                
                bx = bx.to(device)
                
                # 1. Temporal (512-D)
                zt = temp_model.get_representation(bx)
                
                # 2. Morphology (512-D)
                spec = ecg_to_spectrogram(bx)
                zm = morph_model.get_representation(spec)
                
                # 3. Biomarker (32-D)
                bbio = torch.tensor(np.array(batch_bio_feats), dtype=torch.float32).to(device)
                zb = bio_model.encode(bbio)
                
                zt_list.append(zt.cpu().numpy())
                zm_list.append(zm.cpu().numpy())
                zb_list.append(zb.cpu().numpy())
                labels_list.append(by.numpy())
                record_ids_list.extend([int(rid) for rid in batch_records])
                patient_ids_list.extend(batch_patients)
                
        # Concatenate splits
        zt_arr = np.concatenate(zt_list, axis=0)
        zm_arr = np.concatenate(zm_list, axis=0)
        zb_arr = np.concatenate(zb_list, axis=0)
        z_fused_arr = np.concatenate([zt_arr, zm_arr, zb_arr], axis=1) # 1056-D
        labels_arr = np.concatenate(labels_list, axis=0)
        record_ids_arr = np.array(record_ids_list)
        patient_ids_arr = np.array(patient_ids_list)
        
        print(f"  Z_fused shape: {z_fused_arr.shape}")
        
        exported_data[split_name] = {
            "z_fused": z_fused_arr,
            "z_temporal": zt_arr,
            "z_morphology": zm_arr,
            "z_biomarker": zb_arr,
            "labels": labels_arr,
            "record_id": record_ids_arr,
            "patient_id": patient_ids_arr
        }

    # Save 2K benchmark subset
    data_dir = project_root / "data"
    os.makedirs(data_dir, exist_ok=True)
    subset_2k_path = data_dir / "Z_fused_2k.npz"
    print(f"Saving 2K joint dataset to {subset_2k_path}...")
    np.savez_compressed(
        subset_2k_path,
        train_z_fused=exported_data["train"]["z_fused"],
        train_z_temporal=exported_data["train"]["z_temporal"],
        train_z_morphology=exported_data["train"]["z_morphology"],
        train_z_biomarker=exported_data["train"]["z_biomarker"],
        train_labels=exported_data["train"]["labels"],
        train_record_id=exported_data["train"]["record_id"],
        train_patient_id=exported_data["train"]["patient_id"],
        
        val_z_fused=exported_data["val"]["z_fused"],
        val_z_temporal=exported_data["val"]["z_temporal"],
        val_z_morphology=exported_data["val"]["z_morphology"],
        val_z_biomarker=exported_data["val"]["z_biomarker"],
        val_labels=exported_data["val"]["labels"],
        val_record_id=exported_data["val"]["record_id"],
        val_patient_id=exported_data["val"]["patient_id"],
        
        test_z_fused=exported_data["test"]["z_fused"],
        test_z_temporal=exported_data["test"]["z_temporal"],
        test_z_morphology=exported_data["test"]["z_morphology"],
        test_z_biomarker=exported_data["test"]["z_biomarker"],
        test_labels=exported_data["test"]["labels"],
        test_record_id=exported_data["test"]["record_id"],
        test_patient_id=exported_data["test"]["patient_id"]
    )
    
    # Also save it as Z_fused_full.npz so that the rest of our scripts work seamlessly
    full_path = data_dir / "Z_fused_full.npz"
    np.savez_compressed(
        full_path,
        **{k: v for k, v in np.load(subset_2k_path).items()}
    )
    print("Deterministic frozen Z_fused representations exported successfully (2K subset)!")

if __name__ == "__main__":
    main()
