import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import torch
from torch.utils.data import Dataset, DataLoader

class ECGFeatureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.X[idx]  # Return input as target for autoencoders


class BiomarkerPreprocessor:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.label_cols = ["NORM", "MI", "STTC", "CD", "HYP"]

    def load_and_preprocess(self, csv_path: str):
        df = pd.read_csv(csv_path)
        
        # Reconstruct diagnostic_superclasses from scp_codes if missing
        if "diagnostic_superclasses" not in df.columns and "scp_codes" in df.columns:
            import ast
            import os
            scp_csv_path = "data/raw/ptbxl/scp_statements.csv"
            if os.path.exists(scp_csv_path):
                scp_df = pd.read_csv(scp_csv_path, index_col=0)
                code_to_class = scp_df["diagnostic_class"].dropna().to_dict()
                
                def get_classes(scp_str):
                    try:
                        scp_dict = ast.literal_eval(scp_str) if isinstance(scp_str, str) else {}
                        classes = []
                        for code, val in scp_dict.items():
                            if val > 0:
                                mapped = code_to_class.get(code)
                                if mapped:
                                    classes.append(mapped)
                        return str(classes)
                    except Exception:
                        return "[]"
                df["diagnostic_superclasses"] = df["scp_codes"].apply(get_classes)
        
        # Select demographics, HRV, and morphology features as requested
        demographics = ["age", "sex", "height", "weight"]
        hrv_features = [
            "RR_Mean", "RR_Median", "RR_Min", "RR_Max", "RR_Range", "RR_STD", "RR_Variance", "RR_CV", "RR_IQR",
            "RR_Skewness", "RR_Kurtosis", "Mean_HR", "HR_STD", "Min_HR", "Max_HR", "SDNN", "RMSSD", "SDSD", "pNN50",
            "LF_Power", "HF_Power", "LF_HF_Ratio", "SD1", "SD2", "SD1_SD2_Ratio", "Sample_Entropy"
        ]
        
        leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        base_morphology = [
            "PR_Interval", "QRS_Duration", "QT_Interval", "QTc_Bazett", "QTc_Fridericia", "ST_Duration",
            "P_Amplitude", "R_Amplitude", "S_Amplitude", "T_Amplitude", "R_S_Ratio", "QRS_Area", "QRS_Energy",
            "T_wave_Area", "ST_Slope", "QT_Variability", "QT_Dispersion", "Tp_e_Interval", "Tp_e_QT_Ratio",
            "RR_QT_Correlation", "RR_QT_Covariance"
        ]
        
        # Build lead-specific morphology feature names for all 12 leads
        morphology_features = []
        for lead in leads:
            for feat in base_morphology:
                morphology_features.append(f"{lead}_{feat}")
        
        # Fallback to non-prefixed features for compatibility
        for feat in base_morphology:
            morphology_features.append(feat)
        
        # Verify columns exist in the DataFrame
        all_candidate_cols = demographics + hrv_features + morphology_features
        available_cols = []
        for col in all_candidate_cols:
            if col in df.columns:
                available_cols.append(col)
            elif col.replace("_", " ") in df.columns:
                available_cols.append(col.replace("_", " "))
            elif col == "T_wave_Area" and "T_wave_Area" not in df.columns and "T_wave_Area" in df.columns:
                available_cols.append("T_wave_Area")
            else:
                # Search for case-insensitive match or match with different underscores
                matched = False
                for c in df.columns:
                    if c.lower() == col.lower() or c.lower().replace("_", "") == col.lower().replace("_", ""):
                        available_cols.append(c)
                        matched = True
                        break
                if not matched:
                    print(f"Warning: feature column {col} not found in CSV.")
        
        self.feature_cols = available_cols
        
        # Extract features
        X = df[self.feature_cols].copy()
        
        # Create binary mask (1.0 if exists, 0.0 if missing)
        M = (~X.isna()).astype(np.float32).values
        
        # Perform outlier handling (clip to 1st and 99th percentiles)
        for col in X.columns:
            if X[col].dtype in [np.float32, np.float64, np.int32, np.int64]:
                q_low = X[col].quantile(0.01)
                q_high = X[col].quantile(0.99)
                X[col] = np.clip(X[col], q_low, q_high)
        
        # Impute missing values
        X_filled = X.copy()
        for col in X_filled.columns:
            if X_filled[col].isna().all():
                X_filled[col] = 0.0
        X_imputed = self.imputer.fit_transform(X_filled)
        
        # Normalize
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Concatenate scaled features and binary mask
        X_combined = np.hstack([X_scaled, M])
        
        # Map labels for downstream evaluation
        # PTB-XL uses multi-hot diagnostic superclasses.
        # If diagnostic_superclasses contains NORM, MI, etc., we parse them.
        y = np.zeros((len(df), len(self.label_cols)))
        if "diagnostic_superclasses" in df.columns:
            for idx, row in df.iterrows():
                val = str(row["diagnostic_superclasses"])
                for c_idx, label in enumerate(self.label_cols):
                    if label in val:
                        y[idx, c_idx] = 1.0
                        
        return X_combined, y, df

    def get_splits(self, X, y, patient_ids=None):
        """Train/Val/Test split grouped by patient_id to prevent data leakage."""
        if patient_ids is not None:
            unique_patients = np.unique(patient_ids)
            train_patients, test_patients = train_test_split(
                unique_patients, test_size=0.30, random_state=self.random_state
            )
            val_patients, test_patients = train_test_split(
                test_patients, test_size=0.50, random_state=self.random_state
            )
            
            train_idx = np.isin(patient_ids, train_patients)
            val_idx = np.isin(patient_ids, val_patients)
            test_idx = np.isin(patient_ids, test_patients)
            
            return X[train_idx], X[val_idx], X[test_idx], y[train_idx], y[val_idx], y[test_idx]
        else:
            # Fallback to normal split if patient_ids not provided
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.30, random_state=self.random_state
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.50, random_state=self.random_state
            )
            return X_train, X_val, X_test, y_train, y_val, y_test

    def get_dataloaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
        train_dataset = ECGFeatureDataset(X_train, y_train)
        val_dataset = ECGFeatureDataset(X_val, y_val)
        test_dataset = ECGFeatureDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        
        return train_loader, val_loader, test_loader
