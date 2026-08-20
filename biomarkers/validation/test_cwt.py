import os
import sys
import numpy as np
import pandas as pd
import neurokit2 as nk
import wfdb
from pathlib import Path

# Configure paths
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from data_management.loader import PTBXLLoader
from config.config import PTBXL_CONFIG

def main():
    print("Testing DWT vs CWT Delineation on a sample of 100 records...")
    loader = PTBXLLoader(
        root_dir=PTBXL_CONFIG["raw_dir"],
        database_csv=PTBXL_CONFIG["database_csv"],
        scp_csv=PTBXL_CONFIG["scp_csv"],
        resolution="hr"
    )
    metadata = loader.load_metadata()
    sample_ids = metadata.index[:100]
    
    dwt_qrs_durs = []
    cwt_qrs_durs = []
    dwt_pr_ints = []
    cwt_pr_ints = []
    
    dwt_fails = 0
    cwt_fails = 0
    
    for rid in sample_ids:
        row = metadata.loc[rid]
        file_path = loader.root_dir / row["filename_hr"]
        
        try:
            signal, meta = wfdb.rdsamp(str(file_path))
            signal = signal.T.astype(np.float32)
            ii_sig = nk.ecg_clean(signal[1], sampling_rate=500, method="neurokit")
            
            _, info = nk.ecg_peaks(ii_sig, sampling_rate=500, method="neurokit")
            r_peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
            
            if len(r_peaks) < 3:
                continue
                
            # DWT Delineation
            try:
                _, waves_dwt = nk.ecg_delineate(ii_sig, r_peaks, sampling_rate=500, method="dwt")
                d_ons = waves_dwt.get("ECG_R_Onsets", [])
                d_offs = waves_dwt.get("ECG_R_Offsets", [])
                d_dur = np.nanmean([(d_offs[i] - d_ons[i]) * 2.0 for i in range(min(len(d_ons), len(d_offs))) if not (np.isnan(d_ons[i]) or np.isnan(d_offs[i]))])
                dwt_qrs_durs.append(d_dur)
            except Exception:
                dwt_fails += 1
                
            # CWT Delineation
            try:
                _, waves_cwt = nk.ecg_delineate(ii_sig, r_peaks, sampling_rate=500, method="cwt")
                c_ons = waves_cwt.get("ECG_R_Onsets", [])
                c_offs = waves_cwt.get("ECG_R_Offsets", [])
                c_dur = np.nanmean([(c_offs[i] - c_ons[i]) * 2.0 for i in range(min(len(c_ons), len(c_offs))) if not (np.isnan(c_ons[i]) or np.isnan(c_offs[i]))])
                cwt_qrs_durs.append(c_dur)
            except Exception:
                cwt_fails += 1
                
        except Exception as e:
            print(f"Error on record {rid}: {e}")
            
    print(f"\nDWT failed count: {dwt_fails}")
    print(f"CWT failed count: {cwt_fails}")
    print(f"DWT QRS duration mean: {np.nanmean(dwt_qrs_durs):.2f} ms, median: {np.nanmedian(dwt_qrs_durs):.2f} ms")
    print(f"CWT QRS duration mean: {np.nanmean(cwt_qrs_durs):.2f} ms, median: {np.nanmedian(cwt_qrs_durs):.2f} ms")

if __name__ == "__main__":
    main()
