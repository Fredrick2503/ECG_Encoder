import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Configure paths
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from data_management.loader import PTBXLLoader
from config.config import PTBXL_CONFIG
import neurokit2 as nk

FEATURES = [
    "heart_rate", "mean_rr", "sd_rr", "p_amplitude", "p_duration", "pr_interval",
    "v1_r_amplitude", "v1_s_amplitude", "v5_r_amplitude", "max_r_v1_v6",
    "r_progression_slope", "max_st_elevation", "max_st_depression", "num_leads_st_deviation",
    "max_t_amplitude", "mean_t_amplitude", "num_leads_t_inversion", "qrs_duration",
    "qt_interval", "qtc_interval", "qrs_axis", "t_wave_axis", "qrs_t_angle", "sokolow_lyon"
]

def main():
    print("Starting QRS Delineation Audit...")
    
    biomarkers_dir = project_root / "biomarkers"
    val_dir = biomarkers_dir / "validation"
    os.makedirs(val_dir, exist_ok=True)
    
    # Load raw database for statistics
    df_raw = pd.read_csv(biomarkers_dir / "ecg_biomarkers_full.csv")
    
    # 1. Investigate warning frequencies and feature metrics
    print("\n--- Feature Delineation Statistics ---")
    for col_name in ["qrs_duration", "pr_interval", "qt_interval", "qtc_interval", "p_duration"]:
        col = df_raw[col_name]
        missing = col.isna().mean() * 100.0
        # Check clinically suspicious (using normal adult ranges)
        if col_name == "qrs_duration":
            susp = ((col < 60) | (col > 120)).mean() * 100.0
            err = ((col < 40) | (col > 250)).mean() * 100.0
        elif col_name == "pr_interval":
            susp = ((col < 120) | (col > 200)).mean() * 100.0
            err = ((col < 60) | (col > 350)).mean() * 100.0
        elif col_name == "qt_interval" or col_name == "qtc_interval":
            susp = ((col < 350) | (col > 470)).mean() * 100.0
            err = ((col < 200) | (col > 700)).mean() * 100.0
        elif col_name == "p_duration":
            susp = ((col < 60) | (col > 120)).mean() * 100.0
            err = ((col < 30) | (col > 200)).mean() * 100.0
            
        print(f"{col_name.upper()}:")
        print(f"  Missing %: {missing:.2f}%")
        print(f"  Min/Max: {col.min():.2f} / {col.max():.2f}")
        print(f"  Mean/Median: {col.mean():.2f} / {col.median():.2f}")
        print(f"  Clinically Suspicious %: {susp:.2f}%")
        print(f"  Clearly Erroneous/Failed %: {err:.2f}%")

    # 2. Trace QRS on 3 sample records
    print("\n--- Tracing Sample Records ---")
    loader = PTBXLLoader(
        root_dir=PTBXL_CONFIG["raw_dir"],
        database_csv=PTBXL_CONFIG["database_csv"],
        scp_csv=PTBXL_CONFIG["scp_csv"],
        resolution="hr"
    )
    metadata = loader.load_metadata()
    
    # Let's take records: 1, 2, 8
    sample_ids = [1, 2, 8]
    fs = 500
    
    for rid in sample_ids:
        print(f"\nTracing Record {rid}...")
        row = metadata.loc[rid]
        file_path = loader.root_dir / row["filename_hr"]
        
        # Load signal
        import wfdb
        signal, meta = wfdb.rdsamp(str(file_path))
        signal = signal.T.astype(np.float32)
        
        # Lead II signal
        ii_sig = nk.ecg_clean(signal[1], sampling_rate=fs, method="neurokit")
        
        # Peaks
        _, info = nk.ecg_peaks(ii_sig, sampling_rate=fs, method="neurokit")
        r_peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        
        # Delineate using DWT
        _, waves_dwt = nk.ecg_delineate(ii_sig, r_peaks, sampling_rate=fs, method="dwt")
        
        # Delineate using CWT (alternative) for comparison
        try:
            _, waves_cwt = nk.ecg_delineate(ii_sig, r_peaks, sampling_rate=fs, method="cwt")
        except Exception as e:
            waves_cwt = {}
            print(f"  CWT method failed: {e}")
            
        # Get onsets and offsets
        dwt_ons = waves_dwt.get("ECG_R_Onsets", [])
        dwt_offs = waves_dwt.get("ECG_R_Offsets", [])
        cwt_ons = waves_cwt.get("ECG_R_Onsets", [])
        cwt_offs = waves_cwt.get("ECG_R_Offsets", [])
        
        print(f"  Total beats: {len(r_peaks)}")
        
        # Durations (in ms)
        dwt_durs = []
        cwt_durs = []
        for i in range(min(5, len(r_peaks))):
            d_on = dwt_ons[i] if i < len(dwt_ons) else np.nan
            d_off = dwt_offs[i] if i < len(dwt_offs) else np.nan
            c_on = cwt_ons[i] if i < len(cwt_ons) else np.nan
            c_off = cwt_offs[i] if i < len(cwt_offs) else np.nan
            
            d_dur = (d_off - d_on) * 1000.0 / fs if not (np.isnan(d_on) or np.isnan(d_off)) else np.nan
            c_dur = (c_off - c_on) * 1000.0 / fs if not (np.isnan(c_on) or np.isnan(c_off)) else np.nan
            
            dwt_durs.append(d_dur)
            cwt_durs.append(c_dur)
            print(f"    Beat {i}: R-Peak={r_peaks[i]}")
            print(f"      DWT Onset={d_on}, Offset={d_off} -> Dur={d_dur:.2f} ms")
            print(f"      CWT Onset={c_on}, Offset={c_off} -> Dur={c_dur:.2f} ms")
            
        # Save visualization for one beat of Record 1
        if rid == 1:
            beat_idx = 0
            p = r_peaks[beat_idx]
            margin = 150  # samples around peak
            start = max(0, p - margin)
            end = min(len(ii_sig), p + margin)
            
            t = np.arange(start, end) / fs * 1000.0  # relative time in ms
            
            plt.figure(figsize=(10, 6))
            plt.plot(t, ii_sig[start:end], label="Clean Lead II", color="black")
            
            # Draw DWT markers
            if not np.isnan(dwt_ons[beat_idx]):
                plt.axvline(x=dwt_ons[beat_idx]/fs*1000.0, color="red", linestyle="--", label="DWT Onset")
            if not np.isnan(dwt_offs[beat_idx]):
                plt.axvline(x=dwt_offs[beat_idx]/fs*1000.0, color="red", linestyle=":", label="DWT Offset")
                
            # Draw CWT markers
            if beat_idx < len(cwt_ons) and not np.isnan(cwt_ons[beat_idx]):
                plt.axvline(x=cwt_ons[beat_idx]/fs*1000.0, color="blue", linestyle="--", label="CWT Onset")
            if beat_idx < len(cwt_offs) and not np.isnan(cwt_offs[beat_idx]):
                plt.axvline(x=cwt_offs[beat_idx]/fs*1000.0, color="blue", linestyle=":", label="CWT Offset")
                
            plt.axvline(x=p/fs*1000.0, color="green", alpha=0.5, label="R peak")
            plt.title(f"QRS Boundary Comparison (Beat {beat_idx} of Record {rid})")
            plt.xlabel("Time (ms)")
            plt.ylabel("Voltage")
            plt.legend()
            plt.tight_layout()
            plt.savefig(val_dir / f"qrs_trace_comparison_rec_{rid}.png")
            plt.close()
            print(f"Saved trace plot to {val_dir / f'qrs_trace_comparison_rec_{rid}.png'}")

    print("\nTracing complete.")

if __name__ == "__main__":
    main()
