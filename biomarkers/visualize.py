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

def plot_landmarks(record_id, loader):
    """Plot ECG Lead II waveform and overlay P/QRS/T landmarks."""
    record = loader.load_record(record_id)
    raw_signal = record.signal.T  # Transpose to (5000, 12)
    
    # Lead II index is 1
    lead_name = "II"
    sig = raw_signal[:, 1]
    
    # Clean signal
    clean_sig = nk.ecg_clean(sig, sampling_rate=SAMPLING_RATE, method="neurokit")
    
    # Detect peaks and delineate waves
    _, info = nk.ecg_peaks(clean_sig, sampling_rate=SAMPLING_RATE, method="neurokit")
    r_peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    
    if len(r_peaks) < 3:
        logger.warning(f"Record {record_id} has less than 3 R-peaks. Skipping plot.")
        return False
        
    _, waves = nk.ecg_delineate(clean_sig, r_peaks, sampling_rate=SAMPLING_RATE, method="dwt")
    
    # Setup plot
    plt.figure(figsize=(15, 6))
    time_arr = np.arange(len(clean_sig)) / SAMPLING_RATE
    plt.plot(time_arr, clean_sig, label="Filtered Lead II ECG", color="black", linewidth=1.5)
    
    # Plot R peaks
    plt.scatter(time_arr[r_peaks], clean_sig[r_peaks], color="red", marker="o", s=80, zorder=5, label="R Peak")
    
    # Helper to plot markers
    def plot_wave_markers(indices, label, color, marker):
        valid_indices = [int(idx) for idx in indices if not np.isnan(idx) and 0 <= int(idx) < len(clean_sig)]
        if len(valid_indices) > 0:
            plt.scatter(time_arr[valid_indices], clean_sig[valid_indices], color=color, marker=marker, s=60, zorder=5, label=label)

    # Plot wave peaks
    plot_wave_markers(waves.get("ECG_P_Peaks", []), "P Peak", "blue", "^")
    plot_wave_markers(waves.get("ECG_T_Peaks", []), "T Peak", "magenta", "v")
    
    # Plot boundaries as vertical lines or shaded zones
    p_onsets = [int(idx) for idx in waves.get("ECG_P_Onsets", []) if not np.isnan(idx)]
    p_offsets = [int(idx) for idx in waves.get("ECG_P_Offsets", []) if not np.isnan(idx)]
    qrs_onsets = [int(idx) for idx in waves.get("ECG_R_Onsets", []) if not np.isnan(idx)]
    qrs_offsets = [int(idx) for idx in waves.get("ECG_R_Offsets", []) if not np.isnan(idx)]
    t_onsets = [int(idx) for idx in waves.get("ECG_T_Onsets", []) if not np.isnan(idx)]
    t_offsets = [int(idx) for idx in waves.get("ECG_T_Offsets", []) if not np.isnan(idx)]

    # Draw vertical dashed lines for boundaries
    for idx in p_onsets:
        plt.axvline(x=idx/SAMPLING_RATE, color="lightblue", linestyle="--", alpha=0.7)
    for idx in p_offsets:
        plt.axvline(x=idx/SAMPLING_RATE, color="lightblue", linestyle=":", alpha=0.7)
    for idx in qrs_onsets:
        plt.axvline(x=idx/SAMPLING_RATE, color="green", linestyle="--", alpha=0.7)
    for idx in qrs_offsets:
        plt.axvline(x=idx/SAMPLING_RATE, color="green", linestyle=":", alpha=0.7)
    for idx in t_onsets:
        plt.axvline(x=idx/SAMPLING_RATE, color="orange", linestyle="--", alpha=0.7)
    for idx in t_offsets:
        plt.axvline(x=idx/SAMPLING_RATE, color="orange", linestyle=":", alpha=0.7)
        
    # Just draw one line for legend representing the type
    if len(p_onsets) > 0:
        plt.axvline(x=-1, color="lightblue", linestyle="--", label="P Onset/Offset")
    if len(qrs_onsets) > 0:
        plt.axvline(x=-1, color="green", linestyle="--", label="QRS Onset/Offset")
    if len(t_onsets) > 0:
        plt.axvline(x=-1, color="orange", linestyle="--", label="T Onset/Offset")
        
    # Formatting
    plt.title(f"ECG Delineation Landmarks - Record ID: {record_id} (Lead II)", fontsize=14)
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
