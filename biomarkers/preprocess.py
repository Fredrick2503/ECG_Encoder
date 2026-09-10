import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from data_management.loader import PTBXLLoader
from config.config import PTBXL_CONFIG

INPUT_CSV = project_root / "data" / "processed" / "biomarker_features.csv"
OUTPUT_FULL_CSV = project_root / "biomarkers" / "ecg_biomarkers_full.csv"
OUTPUT_PREPROCESSED_CSV = project_root / "biomarkers" / "ecg_biomarkers_preprocessed.csv"
SCALER_PATH = project_root / "biomarkers" / "scaler.pkl"
IMPUTER_PATH = project_root / "biomarkers" / "imputer.pkl"

FEATURES = [
    "RR_Mean", "QRS_Duration", "PR_Interval", "QT_Interval", "QTc_Bazett",
    "ST_Duration", "P_wave_Duration", "R_Amplitude", "P_Amplitude", "T_Amplitude",
    "ST_Deviation", "Q_Amplitude", "R_S_Ratio", "QRS_Energy", "SDNN",
    "RMSSD", "pNN50", "pNN20", "SDRR_RMSSD_Ratio", "HRV_Triangular_Index",
    "LF_Power", "HF_Power", "LF_HF_Ratio", "Total_Power", "Sample_Entropy"
]

LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]

def main():
    print(f"Loading raw biomarker features from {INPUT_CSV}...")
    if not INPUT_CSV.exists():
        print(f"Error: {INPUT_CSV} not found! Please run feature extraction first.")
        sys.exit(1)
        
    df = pd.read_csv(INPUT_CSV)
    
    # Rename ecg_id to record_id
    df.rename(columns={"ecg_id": "record_id"}, inplace=True)
    
    # Load labels
    print("Loading labels from PTB-XL metadata...")
    loader = PTBXLLoader(
        root_dir=PTBXL_CONFIG["raw_dir"],
        database_csv=PTBXL_CONFIG["database_csv"],
        scp_csv=PTBXL_CONFIG["scp_csv"],
        resolution="hr"
    )
    metadata = loader.load_metadata()
    
    print("Mapping labels...")
    label_cols = {lbl: [] for lbl in LABELS}
    for record_id in df["record_id"]:
        if record_id in metadata.index:
            row = metadata.loc[record_id]
            scp_codes = row.get("scp_codes", {})
            diagnostic_classes = loader.parser.get_diagnostic_classes(scp_codes)
            for lbl in LABELS:
                label_cols[lbl].append(1 if lbl in diagnostic_classes else 0)
        else:
            for lbl in LABELS:
                label_cols[lbl].append(0)
                
    for lbl in LABELS:
        df[lbl] = label_cols[lbl]
        
    # Save the full raw csv (raw features + labels)
    print(f"Saving raw full biomarker features with labels to {OUTPUT_FULL_CSV}...")
    df.to_csv(OUTPUT_FULL_CSV, index=False)
    
    # 1. Identify missing/NaN values and impute using median
    print("Fitting imputer (calculating dataset medians)...")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_raw = df[FEATURES].copy()
    X_imputed = imputer.fit_transform(X_raw)
    
    # 2. Standardize features using StandardScaler
    print("Fitting scaler (Standardizing to mean=0, std=1)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # 3. Save parameters using pickle
    print(f"Saving scaler to {SCALER_PATH}...")
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
        
    print(f"Saving imputer to {IMPUTER_PATH}...")
    with open(IMPUTER_PATH, "wb") as f:
        pickle.dump(imputer, f)
        
    # 4. Create new DataFrame for processed feature matrix
    df_preprocessed = pd.DataFrame(X_scaled, columns=FEATURES)
    df_preprocessed.insert(0, "record_id", df["record_id"].values)
    
    # Append class labels
    for lbl in LABELS:
        df_preprocessed[lbl] = df[lbl].values
        
    # Save CSV
    print(f"Saving processed features to {OUTPUT_PREPROCESSED_CSV}...")
    df_preprocessed.to_csv(OUTPUT_PREPROCESSED_CSV, index=False)
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
