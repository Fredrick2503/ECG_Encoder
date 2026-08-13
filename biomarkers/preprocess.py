import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Paths
project_root = Path(__file__).resolve().parent.parent
INPUT_CSV = project_root / "biomarkers" / "ecg_biomarkers_4500.csv"
OUTPUT_CSV = project_root / "biomarkers" / "ecg_biomarkers_preprocessed.csv"
SCALER_PATH = project_root / "biomarkers" / "scaler.pkl"
IMPUTER_PATH = project_root / "biomarkers" / "imputer.pkl"

FEATURES = [
    "heart_rate", "mean_rr", "sd_rr", "p_amplitude", "p_duration", "pr_interval",
    "v1_r_amplitude", "v1_s_amplitude", "v5_r_amplitude", "max_r_v1_v6",
    "r_progression_slope", "max_st_elevation", "max_st_depression", "num_leads_st_deviation",
    "max_t_amplitude", "mean_t_amplitude", "num_leads_t_inversion", "qrs_duration",
    "qt_interval", "qtc_interval", "qrs_axis", "t_wave_axis", "qrs_t_angle", "sokolow_lyon"
]

LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]

def main():
    print(f"Loading raw biomarker features from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # 1. Identify missing/NaN values and impute using median
    imputer = SimpleImputer(strategy="median")
    X_raw = df[FEATURES].copy()
    
    print("Fitting imputer (calculating dataset medians)...")
    X_imputed = imputer.fit_transform(X_raw)
    
    # 2. Standardize features using StandardScaler
    scaler = StandardScaler()
    print("Fitting scaler (Standardizing to mean=0, std=1)...")
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
    print(f"Saving processed features to {OUTPUT_CSV}...")
    df_preprocessed.to_csv(OUTPUT_CSV, index=False)
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
