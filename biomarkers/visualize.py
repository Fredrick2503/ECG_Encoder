import os
import sys
import logging
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ECGBiomarkerVisualization")

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from data_management.loader import PTBXLLoader
from config.config import PTBXL_CONFIG

# Settings
NUM_PLOTS = 10
SAMPLING_RATE = 500
OUTPUT_DIR = project_root / "biomarkers" / "validation_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 12 Standard Lead mapping for PTB-XL
LEADS_MAP = {
    "I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5,
    "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11
}

def plot_landmarks(record_id, loader):
    """Plot ECG Lead II waveform and overlay P/QRS/T landmarks using improved multi-lead logic."""
    record = loader.load_record(record_id)
    raw_signal = record.signal.T  # Transpose to (5000, 12)
    
    # 1. Clean Lead II for plotting
    ii_idx = LEADS_MAP["II"]
    ii_clean = nk.ecg_clean(raw_signal[:, ii_idx], sampling_rate=SAMPLING_RATE, method="neurokit")
    
    # Clean fallback leads
    clean_signals = {}
    for lead in ["II", "V5", "V1", "I"]:
        clean_signals[lead] = nk.ecg_clean(raw_signal[:, LEADS_MAP[lead]], sampling_rate=SAMPLING_RATE, method="neurokit")

    # 2. Detect R-peaks on Lead II
    try:
        _, info = nk.ecg_peaks(ii_clean, sampling_rate=SAMPLING_RATE, method="neurokit")
        r_peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    except Exception:
        return False
        
    if len(r_peaks) < 3:
        logger.warning(f"Record {record_id} has less than 3 R-peaks. Skipping plot.")
        return False

    # 3. Delineate subset of leads (II, V5, V1, I) using DWT
    delineations = {}
    for lead in ["II", "V5", "V1", "I"]:
        try:
            _, waves = nk.ecg_delineate(clean_signals[lead], r_peaks, sampling_rate=SAMPLING_RATE, method="dwt")
            delineations[lead] = waves
        except Exception:
            delineations[lead] = {}

    # 4. Reliable P-wave lead selection (II -> V1 -> I)
    p_lead = "II"
    p_waves = delineations.get("II", {})
    p_onsets_ii = p_waves.get("ECG_P_Onsets", [])
    if hasattr(p_onsets_ii, "tolist"):
        p_onsets_ii = p_onsets_ii.tolist()
    valid_p_ii = [p for p in p_onsets_ii if not pd.isna(p)]
    
    if len(valid_p_ii) < 0.6 * len(r_peaks):
        p_waves_v1 = delineations.get("V1", {})
        p_onsets_v1 = p_waves_v1.get("ECG_P_Onsets", [])
        if hasattr(p_onsets_v1, "tolist"):
            p_onsets_v1 = p_onsets_v1.tolist()
        valid_p_v1 = [p for p in p_onsets_v1 if not pd.isna(p)]
        if len(valid_p_v1) > len(valid_p_ii):
            p_lead = "V1"
            p_waves = p_waves_v1
            
    p_onsets_curr = p_waves.get("ECG_P_Onsets", [])
    if hasattr(p_onsets_curr, "tolist"):
        p_onsets_curr = p_onsets_curr.tolist()
    valid_p_curr = [p for p in p_onsets_curr if not pd.isna(p)]
    if len(valid_p_curr) < 0.6 * len(r_peaks):
        p_waves_i = delineations.get("I", {})
        p_onsets_i = p_waves_i.get("ECG_P_Onsets", [])
        if hasattr(p_onsets_i, "tolist"):
            p_onsets_i = p_onsets_i.tolist()
        valid_p_i = [p for p in p_onsets_i if not pd.isna(p)]
        if len(valid_p_i) > len(valid_p_curr):
            p_lead = "I"
            p_waves = p_waves_i

    # 5. Reliable T-wave lead selection (II -> V5 -> I)
    t_lead = "II"
    t_waves = delineations.get("II", {})
    t_offsets_ii = t_waves.get("ECG_T_Offsets", [])
    if hasattr(t_offsets_ii, "tolist"):
        t_offsets_ii = t_offsets_ii.tolist()
    valid_t_ii = [t for t in t_offsets_ii if not pd.isna(t)]
    
    if len(valid_t_ii) < 0.6 * len(r_peaks):
        t_waves_v5 = delineations.get("V5", {})
        t_offsets_v5 = t_waves_v5.get("ECG_T_Offsets", [])
        if hasattr(t_offsets_v5, "tolist"):
            t_offsets_v5 = t_offsets_v5.tolist()
        valid_t_v5 = [t for t in t_offsets_v5 if not pd.isna(t)]
        if len(valid_t_v5) > len(valid_t_ii):
            t_lead = "V5"
            t_waves = t_waves_v5
            
    t_offsets_curr = t_waves.get("ECG_T_Offsets", [])
    if hasattr(t_offsets_curr, "tolist"):
        t_offsets_curr = t_offsets_curr.tolist()
    valid_t_curr = [t for t in t_offsets_curr if not pd.isna(t)]
    if len(valid_t_curr) < 0.6 * len(r_peaks):
        t_waves_i = delineations.get("I", {})
        t_offsets_i = t_waves_i.get("ECG_T_Offsets", [])
        if hasattr(t_offsets_i, "tolist"):
            t_offsets_i = t_offsets_i.tolist()
        valid_t_i = [t for t in t_offsets_i if not pd.isna(t)]
        if len(valid_t_i) > len(valid_t_curr):
            t_lead = "I"
            t_waves = t_waves_i

    # 6. Global QRS boundaries across II, V5, V1, I
    lead_onsets = []
    lead_offsets = []
    for l_name in ["II", "V5", "V1", "I"]:
        w = delineations.get(l_name, {})
        ons = w.get("ECG_R_Onsets", [])
        offs = w.get("ECG_R_Offsets", [])
        if hasattr(ons, "tolist"):
            ons = ons.tolist()
        if hasattr(offs, "tolist"):
            offs = offs.tolist()
        lead_onsets.append(ons)
        lead_offsets.append(offs)
        
    global_qrs_onsets = []
    global_qrs_offsets = []
    for beat_idx in range(len(r_peaks)):
        beat_onsets = []
        beat_offsets = []
        for l_idx in range(4):
            ons_list = lead_onsets[l_idx]
            offs_list = lead_offsets[l_idx]
            if beat_idx < len(ons_list) and not pd.isna(ons_list[beat_idx]):
                beat_onsets.append(ons_list[beat_idx])
            if beat_idx < len(offs_list) and not pd.isna(offs_list[beat_idx]):
                beat_offsets.append(offs_list[beat_idx])
        if beat_onsets and beat_offsets:
            global_qrs_onsets.append(int(np.median(beat_onsets)))
            global_qrs_offsets.append(int(np.median(beat_offsets)))

    # Setup plot
    plt.figure(figsize=(15, 6))
    time_arr = np.arange(len(ii_clean)) / SAMPLING_RATE
    plt.plot(time_arr, ii_clean, label="Filtered Lead II ECG", color="black", linewidth=1.5)
    
    # Plot R peaks
    plt.scatter(time_arr[r_peaks], ii_clean[r_peaks], color="red", marker="o", s=80, zorder=5, label="R Peak")
    
    # Helper to plot markers
    def plot_wave_markers(indices, label, color, marker):
        valid_indices = [int(idx) for idx in indices if not np.isnan(idx) and 0 <= int(idx) < len(ii_clean)]
        if len(valid_indices) > 0:
            plt.scatter(time_arr[valid_indices], ii_clean[valid_indices], color=color, marker=marker, s=60, zorder=5, label=label)

    # Plot wave peaks (P peak from p_lead, T peak from t_lead)
    plot_wave_markers(p_waves.get("ECG_P_Peaks", []), f"P Peak ({p_lead})", "blue", "^")
    plot_wave_markers(t_waves.get("ECG_T_Peaks", []), f"T Peak ({t_lead})", "magenta", "v")
    
    # Plot boundaries
    p_onsets = [int(idx) for idx in p_waves.get("ECG_P_Onsets", []) if not np.isnan(idx)]
    p_offsets = [int(idx) for idx in p_waves.get("ECG_P_Offsets", []) if not np.isnan(idx)]
    t_onsets = [int(idx) for idx in t_waves.get("ECG_T_Onsets", []) if not np.isnan(idx)]
    t_offsets = [int(idx) for idx in t_waves.get("ECG_T_Offsets", []) if not np.isnan(idx)]

    # Draw vertical dashed lines for boundaries
    for idx in p_onsets:
        plt.axvline(x=idx/SAMPLING_RATE, color="lightblue", linestyle="--", alpha=0.7)
    for idx in p_offsets:
        plt.axvline(x=idx/SAMPLING_RATE, color="lightblue", linestyle=":", alpha=0.7)
    for idx in global_qrs_onsets:
        plt.axvline(x=idx/SAMPLING_RATE, color="green", linestyle="--", alpha=0.7)
    for idx in global_qrs_offsets:
        plt.axvline(x=idx/SAMPLING_RATE, color="green", linestyle=":", alpha=0.7)
    for idx in t_onsets:
        plt.axvline(x=idx/SAMPLING_RATE, color="orange", linestyle="--", alpha=0.7)
    for idx in t_offsets:
        plt.axvline(x=idx/SAMPLING_RATE, color="orange", linestyle=":", alpha=0.7)
        
    # Legend lines representation
    if len(p_onsets) > 0:
        plt.axvline(x=-1, color="lightblue", linestyle="--", label=f"P Bound ({p_lead})")
    if len(global_qrs_onsets) > 0:
        plt.axvline(x=-1, color="green", linestyle="--", label="Global QRS Bound")
    if len(t_onsets) > 0:
        plt.axvline(x=-1, color="orange", linestyle="--", label=f"T Bound ({t_lead})")
        
    # Formatting
    plt.title(f"ECG Delineation Landmarks - Record ID: {record_id} (Lead II Display)", fontsize=14)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Amplitude (mV)", fontsize=12)
    plt.xlim(0, 10.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    
    # Save plot
    save_path = OUTPUT_DIR / f"landmarks_record_{record_id}.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Saved visualization plot for record {record_id} to {save_path}")
    return True

def main():
    logger.info("Initializing dataset loader to select records for visualization...")
    loader = PTBXLLoader(
        root_dir=PTBXL_CONFIG["raw_dir"],
        database_csv=PTBXL_CONFIG["database_csv"],
        scp_csv=PTBXL_CONFIG["scp_csv"],
        resolution="hr"
    )
    metadata = loader.load_metadata()
    
    # Select 10 random records
    random.seed(42)
    all_record_ids = list(metadata.index)
    selected_ids = random.sample(all_record_ids, NUM_PLOTS * 2)  # sample a few extra in case of failures
    
    plotted = 0
    for rid in selected_ids:
        if plotted >= NUM_PLOTS:
            break
        success = plot_landmarks(rid, loader)
        if success:
            plotted += 1

    logger.info(f"Successfully generated validation plots for {plotted} records in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
