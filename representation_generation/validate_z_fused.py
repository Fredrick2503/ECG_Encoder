import os
import sys
import numpy as np
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def run_validation(data_path: Path):
    print(f"==================================================")
    print(f"Validating dataset: {data_path.name}")
    print(f"==================================================")
    
    if not data_path.exists():
        print(f"Error: Dataset path {data_path} does not exist.")
        return False
        
    data = np.load(data_path)
    
    # Check splits
    splits = ["train", "val", "test"]
    all_ok = True
    
    patient_sets = {}
    
    for split in splits:
        print(f"\n--- Checking split: {split} ---")
        
        # 1. Key existence checks
        keys = [f"{split}_z_fused", f"{split}_z_temporal", f"{split}_z_morphology", 
                f"{split}_z_biomarker", f"{split}_labels", f"{split}_record_id", f"{split}_patient_id"]
        for k in keys:
            if k not in data:
                print(f"  [FAIL] Missing key: {k}")
                all_ok = False
                continue
                
        if not all_ok:
            continue
            
        z_f = data[f"{split}_z_fused"]
        z_t = data[f"{split}_z_temporal"]
        z_m = data[f"{split}_z_morphology"]
        z_b = data[f"{split}_z_biomarker"]
        labels = data[f"{split}_labels"]
        record_ids = data[f"{split}_record_id"]
        patient_ids = data[f"{split}_patient_id"]
        
        N = len(record_ids)
        print(f"  Number of records: {N}")
        
        # 2. Dimensions Validation
        dims_ok = True
        if z_f.shape != (N, 1056):
            print(f"  [FAIL] z_fused shape mismatch: {z_f.shape} (expected ({N}, 1056))")
            dims_ok = False
        if z_t.shape != (N, 512):
            print(f"  [FAIL] z_temporal shape mismatch: {z_t.shape} (expected ({N}, 512))")
            dims_ok = False
        if z_m.shape != (N, 512):
            print(f"  [FAIL] z_morphology shape mismatch: {z_m.shape} (expected ({N}, 512))")
            dims_ok = False
        if z_b.shape != (N, 32):
            print(f"  [FAIL] z_biomarker shape mismatch: {z_b.shape} (expected ({N}, 32))")
            dims_ok = False
        if labels.shape != (N, 5):
            print(f"  [FAIL] labels shape mismatch: {labels.shape} (expected ({N}, 5))")
            dims_ok = False
        if record_ids.shape != (N,):
            print(f"  [FAIL] record_ids shape mismatch: {record_ids.shape} (expected ({N},))")
            dims_ok = False
        if patient_ids.shape != (N,):
            print(f"  [FAIL] patient_ids shape mismatch: {patient_ids.shape} (expected ({N},))")
            dims_ok = False
            
        if dims_ok:
            print("  [PASS] Dimensions and shapes verified.")
        else:
            all_ok = False
            
        # 3. Concatenation / Alignment validation
        concat_check = np.concatenate([z_t, z_m, z_b], axis=1)
        if not np.allclose(z_f, concat_check, atol=1e-6):
            print("  [FAIL] z_fused is not exactly the concatenation of temporal + morphology + biomarker representations!")
            all_ok = False
        else:
            print("  [PASS] Concatenation reconstruction checked.")

        # 4. NaN / Inf validation
        for name, arr in [("z_fused", z_f), ("labels", labels), ("record_id", record_ids), ("patient_id", patient_ids)]:
            if np.isnan(arr).any():
                print(f"  [FAIL] {name} contains NaNs.")
                all_ok = False
            elif np.isinf(arr).any():
                print(f"  [FAIL] {name} contains Inf values.")
                all_ok = False
            else:
                pass
        print("  [PASS] NaN / Inf check passed.")

        # 5. Duplicates validation
        u_records, counts = np.unique(record_ids, return_counts=True)
        if len(u_records) != len(record_ids):
            print(f"  [FAIL] Duplicate record_ids detected in split {split}. Count of unique={len(u_records)}, total={len(record_ids)}")
            all_ok = False
        else:
            print("  [PASS] No duplicate record IDs found.")
            
        # Store patient IDs (excluding -1 for missing values)
        p_ids_valid = set(patient_ids[patient_ids != -1].tolist())
        patient_sets[split] = p_ids_valid

    # 6. Patient Leakage Validation
    print("\n--- Checking patient-wise split leakage ---")
    splits_pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    leakage_detected = False
    for s1, s2 in splits_pairs:
        intersect = patient_sets[s1].intersection(patient_sets[s2])
        if len(intersect) > 0:
            print(f"  [FAIL] Leakage detected! {len(intersect)} patients overlap between {s1} and {s2} splits.")
            print(f"  Overlapping patient IDs (first 10): {list(intersect)[:10]}")
            leakage_detected = True
            all_ok = False
        else:
            print(f"  [PASS] Zero patient ID leakage between {s1} and {s2} splits.")
            
    if all_ok:
        print("\n[SUCCESS] All integrity checks passed successfully!")
    else:
        print("\n[FAILURE] Some integrity checks failed. Check output logs above.")
        
    return all_ok

def main():
    data_dir = project_root / "data"
    full_path = data_dir / "Z_fused_full.npz"
    subset_path = data_dir / "Z_fused_2k.npz"
    
    full_ok = run_validation(full_path)
    subset_ok = run_validation(subset_path)
    
    if full_ok and subset_ok:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
