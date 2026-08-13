import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import scipy.signal
import neurokit2 as nk
from pathlib import Path
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ECGBiomarkerExtraction")

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from data_management.loader import PTBXLLoader
from config.config import PTBXL_CONFIG

# Configuration parameters
ROW_LIMIT = 4500  # Process the first 4500 records
SAMPLING_RATE = 500
FILTER_LOWCUT = 0.5
FILTER_HIGHCUT = 40.0
OUTPUT_DIR = project_root / "biomarkers"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_OUTPUT_PATH = OUTPUT_DIR / "ecg_biomarkers_4500.csv"
REPORT_OUTPUT_PATH = OUTPUT_DIR / "extraction_report.txt"

# 12 Standard Lead mapping for PTB-XL
LEADS_MAP = {
    "I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5,
    "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11
}

def clean_signal(signal_1d, fs=SAMPLING_RATE):
    """Filter signal using a bandpass filter (0.5 to 40 Hz)."""
    return nk.ecg_clean(signal_1d, sampling_rate=fs, method="neurokit")

def get_isoelectric_baseline(ecg_clean, r_peaks, waves, fs=SAMPLING_RATE):
    """Estimate isoelectric baseline using the average PR segment or PQ junctions."""
    baselines = []
    p_offsets = waves.get("ECG_P_Offsets", [])
    qrs_onsets = waves.get("ECG_R_Onsets", [])
    
    if hasattr(p_offsets, "tolist"):
        p_offsets = p_offsets.tolist()
    if hasattr(qrs_onsets, "tolist"):
        qrs_onsets = qrs_onsets.tolist()
        
    for i in range(len(r_peaks)):
        p_off = p_offsets[i] if i < len(p_offsets) else np.nan
        q_on = qrs_onsets[i] if i < len(qrs_onsets) else np.nan
        
        # If PR interval boundaries exist, average the signal between them
        if not pd.isna(p_off) and not pd.isna(q_on) and q_on > p_off:
            segment = ecg_clean[int(p_off):int(q_on)]
            if len(segment) > 0:
                baselines.append(np.mean(segment))
                continue
                
        # Fallback: take average of 50ms to 10ms window before the R-peak/QRS onset
        q_start = q_on if not pd.isna(q_on) else r_peaks[i]
        start_idx = int(q_start - 0.05 * fs)
        end_idx = int(q_start - 0.01 * fs)
        if start_idx >= 0 and end_idx < len(ecg_clean):
            baselines.append(np.mean(ecg_clean[start_idx:end_idx]))
            
    return np.median(baselines) if len(baselines) > 0 else 0.0

def calculate_axis(lead_I_val, lead_aVF_val):
    """Calculate the electrical axis in the frontal plane (degrees)."""
    if pd.isna(lead_I_val) or pd.isna(lead_aVF_val):
        return np.nan
    angle = np.arctan2(lead_aVF_val, lead_I_val) * 180.0 / np.pi
    return float(angle)

def calculate_qrs_t_angle(qrs_axis, t_axis):
    """Calculate the spatial QRS-T angle wrapped to [0, 180] degrees."""
    if pd.isna(qrs_axis) or pd.isna(t_axis):
        return np.nan
    angle = abs(qrs_axis - t_axis)
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)

def extract_record_features(record, fs=SAMPLING_RATE):
    """Extract clinical biomarkers and perform QC checks on a single record."""
    raw_signal = record.signal.T  # Transpose to (time_steps, channels) -> (5000, 12)
    record_id = record.record_id
    
    qc_failures = []
    
    # 1. Clean all signals
    clean_signals = np.zeros_like(raw_signal)
    for lead_name, col_idx in LEADS_MAP.items():
        clean_signals[:, col_idx] = clean_signal(raw_signal[:, col_idx], fs)
        
    # 2. Detect R-peaks on Lead II (used as global time reference since simultaneous)
    ii_clean = clean_signals[:, LEADS_MAP["II"]]
    try:
        _, info = nk.ecg_peaks(ii_clean, sampling_rate=fs, method="neurokit")
        ii_r_peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    except Exception as e:
        return None, [f"R-peak detection failed: {str(e)}"]

    if len(ii_r_peaks) < 3:
        return None, ["ECG unusable: less than 3 R-peaks detected in Lead II."]

    # 3. Delineate subset of leads (II, V5, V1, I) using DWT
    delineations = {}
    for lead in ["II", "V5", "V1", "I"]:
        lead_sig = clean_signals[:, LEADS_MAP[lead]]
        try:
            _, waves = nk.ecg_delineate(lead_sig, ii_r_peaks, sampling_rate=fs, method="dwt")
            delineations[lead] = waves
        except Exception as e:
            delineations[lead] = {}
            qc_failures.append(f"{lead}_delineation_error: {str(e)}")

    # 4. Reliable P-wave lead selection (II -> V1 -> I)
    p_lead = "II"
    p_waves = delineations.get("II", {})
    p_onsets_ii = p_waves.get("ECG_P_Onsets", [])
    if hasattr(p_onsets_ii, "tolist"):
        p_onsets_ii = p_onsets_ii.tolist()
    valid_p_ii = [p for p in p_onsets_ii if not pd.isna(p)]
    
    if len(valid_p_ii) < 0.6 * len(ii_r_peaks):
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
    if len(valid_p_curr) < 0.6 * len(ii_r_peaks):
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
    
    if len(valid_t_ii) < 0.6 * len(ii_r_peaks):
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
    if len(valid_t_curr) < 0.6 * len(ii_r_peaks):
        t_waves_i = delineations.get("I", {})
        t_offsets_i = t_waves_i.get("ECG_T_Offsets", [])
        if hasattr(t_offsets_i, "tolist"):
            t_offsets_i = t_offsets_i.tolist()
        valid_t_i = [t for t in t_offsets_i if not pd.isna(t)]
        if len(valid_t_i) > len(valid_t_curr):
            t_lead = "I"
            t_waves = t_waves_i

    # 6. Baseline estimation per lead
    lead_baselines = {}
    for lead_name in LEADS_MAP.keys():
        clean_sig = clean_signals[:, LEADS_MAP[lead_name]]
        if lead_name in delineations:
            lead_baselines[lead_name] = get_isoelectric_baseline(clean_sig, ii_r_peaks, delineations[lead_name], fs)
        else:
            lead_baselines[lead_name] = get_isoelectric_baseline(clean_sig, ii_r_peaks, {}, fs)

    # Initialize feature dict
    feats = {"record_id": record_id}
    
    # Helper to convert samples to ms
    def samples_to_ms(samples):
        return float(samples * 1000.0 / fs) if not np.isnan(samples) else np.nan

    # Helper to get average amplitude relative to baseline
    def get_avg_amp(lead, peak_indices, val_multiplier=1.0):
        if peak_indices is None:
            return np.nan
        if hasattr(peak_indices, "tolist"):
            peak_indices = peak_indices.tolist()
        elif isinstance(peak_indices, np.ndarray):
            peak_indices = list(peak_indices)
        
        peak_indices = [idx for idx in peak_indices if not pd.isna(idx)]
        if len(peak_indices) == 0:
            return np.nan
            
        clean_sig = clean_signals[:, LEADS_MAP[lead]]
        base = lead_baselines[lead]
        amps = []
        for idx in peak_indices:
            if 0 <= int(idx) < len(clean_sig):
                amps.append(clean_sig[int(idx)] - base)
        return float(np.mean(amps)) * val_multiplier if len(amps) > 0 else np.nan

    # Helper to get average duration
    def get_avg_duration(start_indices, end_indices):
        if start_indices is None or end_indices is None:
            return np.nan
        if hasattr(start_indices, "tolist"):
            start_indices = start_indices.tolist()
        if hasattr(end_indices, "tolist"):
            end_indices = end_indices.tolist()
            
        durs = []
        for s, e in zip(start_indices, end_indices):
            if not pd.isna(s) and not pd.isna(e) and e > s:
                durs.append(e - s)
        return samples_to_ms(np.mean(durs)) if len(durs) > 0 else np.nan

    # =========================================================================
    # 1. Lead II Features
    # =========================================================================
    rr_intervals = np.diff(ii_r_peaks)
    rr_ms = [samples_to_ms(rr) for rr in rr_intervals]
    
    feats["heart_rate"] = float(60.0 / (np.mean(rr_intervals) / fs)) if len(rr_intervals) > 0 else np.nan
    feats["mean_rr"] = float(np.mean(rr_ms)) if len(rr_ms) > 0 else np.nan
    feats["sd_rr"] = float(np.std(rr_ms)) if len(rr_ms) > 0 else np.nan
    
    feats["p_amplitude"] = get_avg_amp(p_lead, p_waves.get("ECG_P_Peaks", []))
    feats["p_duration"] = get_avg_duration(p_waves.get("ECG_P_Onsets", []), p_waves.get("ECG_P_Offsets", []))
    feats["pr_interval"] = get_avg_duration(p_waves.get("ECG_P_Onsets", []), p_waves.get("ECG_R_Onsets", []))

    # =========================================================================
    # 2. Lead V1 Features
    # =========================================================================
    v1_clean = clean_signals[:, LEADS_MAP["V1"]]
    v1_base = lead_baselines["V1"]
    
    feats["v1_r_amplitude"] = get_avg_amp("V1", ii_r_peaks)
    
    s_amps = []
    for r_peak in ii_r_peaks:
        start = int(r_peak)
        end = min(len(v1_clean), int(r_peak + 0.08 * fs))
        if start < end:
            s_amps.append(np.min(v1_clean[start:end]) - v1_base)
    feats["v1_s_amplitude"] = float(np.mean(s_amps)) if len(s_amps) > 0 else np.nan

    # =========================================================================
    # 3. Lead V5 Features
    # =========================================================================
    feats["v5_r_amplitude"] = get_avg_amp("V5", ii_r_peaks)

    # =========================================================================
    # 4. V1–V6 R-wave Progression
    # =========================================================================
    v_leads = ["V1", "V2", "V3", "V4", "V5", "V6"]
    v_r_amps = [get_avg_amp(l, ii_r_peaks) for l in v_leads]
    feats["max_r_v1_v6"] = float(np.nanmax(v_r_amps)) if not np.all(np.isnan(v_r_amps)) else np.nan
    
    valid_r = [(idx, val) for idx, val in enumerate(v_r_amps) if not np.isnan(val)]
    if len(valid_r) >= 2:
        x_vals = [item[0] for item in valid_r]
        y_vals = [item[1] for item in valid_r]
        slope, _ = np.polyfit(x_vals, y_vals, 1)
        feats["r_progression_slope"] = float(slope)
    else:
        feats["r_progression_slope"] = np.nan

    # =========================================================================
    # 5. All 12 Leads Metrics (ST and T-wave properties)
    # =========================================================================
    # Find average QRS offset relative to R-peak from Lead II
    ii_qrs_offsets = delineations.get("II", {}).get("ECG_R_Offsets", [])
    relative_j_offsets = []
    if hasattr(ii_qrs_offsets, "tolist"):
        ii_qrs_offsets = ii_qrs_offsets.tolist()
    for r_peak, qrs_off in zip(ii_r_peaks, ii_qrs_offsets):
        if not pd.isna(r_peak) and not pd.isna(qrs_off):
            relative_j_offsets.append(qrs_off - r_peak)
    avg_j_offset = np.mean(relative_j_offsets) if len(relative_j_offsets) > 0 else 0.08 * fs
    
    st_elevations = []
    st_depressions = []
    num_significant_st = 0
    t_amplitudes = []
    num_t_inversions = 0
    
    for lead_name, col_idx in LEADS_MAP.items():
        clean_sig = clean_signals[:, col_idx]
        base = lead_baselines[lead_name]
        
        st_vals = []
        for r_peak in ii_r_peaks:
            j_point = int(r_peak + avg_j_offset)
            st_start = int(j_point + 0.06 * fs)
            st_end = int(j_point + 0.08 * fs)
            if st_start < len(clean_sig) and st_end < len(clean_sig):
                st_vals.append(np.mean(clean_sig[st_start:st_end]) - base)
                
        lead_st_mean = np.mean(st_vals) if len(st_vals) > 0 else np.nan
        if not np.isnan(lead_st_mean):
            if lead_st_mean > 0:
                st_elevations.append(lead_st_mean)
            else:
                st_depressions.append(abs(lead_st_mean))
            if abs(lead_st_mean) >= 0.1:
                num_significant_st += 1
                
        lead_t_vals = []
        for r_peak in ii_r_peaks:
            start = int(r_peak + 0.10 * fs)
            end = min(len(clean_sig), int(r_peak + 0.40 * fs))
            if start < end:
                segment = clean_sig[start:end] - base
                max_idx = np.argmax(np.abs(segment))
                lead_t_vals.append(segment[max_idx])
                
        lead_t_mean = np.mean(lead_t_vals) if len(lead_t_vals) > 0 else np.nan
        if not np.isnan(lead_t_mean):
            t_amplitudes.append(abs(lead_t_mean))
            if lead_t_mean < 0.0:
                num_t_inversions += 1
                
    feats["max_st_elevation"] = float(np.max(st_elevations)) if len(st_elevations) > 0 else 0.0
    feats["max_st_depression"] = float(np.max(st_depressions)) if len(st_depressions) > 0 else 0.0
    feats["num_leads_st_deviation"] = num_significant_st
    feats["max_t_amplitude"] = float(np.max(t_amplitudes)) if len(t_amplitudes) > 0 else np.nan
    feats["mean_t_amplitude"] = float(np.mean(t_amplitudes)) if len(t_amplitudes) > 0 else np.nan
    feats["num_leads_t_inversion"] = num_t_inversions

    # =========================================================================
    # 6. Global / 12-lead ECG (QRS, QT, QTc)
    # =========================================================================
    # Global QRS boundaries across II, V5, V1, I
    lead_onsets = []
    lead_offsets = []
    for lead in ["II", "V5", "V1", "I"]:
        w = delineations.get(lead, {})
        ons = w.get("ECG_R_Onsets", [])
        offs = w.get("ECG_R_Offsets", [])
        if hasattr(ons, "tolist"):
            ons = ons.tolist()
        if hasattr(offs, "tolist"):
            offs = offs.tolist()
        lead_onsets.append(ons)
        lead_offsets.append(offs)
        
    qrs_durs = []
    for beat_idx in range(len(ii_r_peaks)):
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
            global_onset = np.median(beat_onsets)
            global_offset = np.median(beat_offsets)
            if global_offset > global_onset:
                qrs_durs.append(global_offset - global_onset)
                
    feats["qrs_duration"] = samples_to_ms(np.mean(qrs_durs)) if len(qrs_durs) > 0 else np.nan
    
    # QT interval (from global QRS onset to reliable T offset)
    qrs_onsets = t_waves.get("ECG_R_Onsets", [])
    t_offsets = t_waves.get("ECG_T_Offsets", [])
    if hasattr(qrs_onsets, "tolist"):
        qrs_onsets = qrs_onsets.tolist()
    if hasattr(t_offsets, "tolist"):
        t_offsets = t_offsets.tolist()
        
    qt_intervals = []
    for q_on, t_off in zip(qrs_onsets, t_offsets):
        if not pd.isna(q_on) and not pd.isna(t_off) and t_off > q_on:
            qt_intervals.append(t_off - q_on)
    feats["qt_interval"] = samples_to_ms(np.mean(qt_intervals)) if len(qt_intervals) > 0 else np.nan
    
    # QTc Fridericia
    if not np.isnan(feats["qt_interval"]) and not np.isnan(feats["mean_rr"]):
        rr_sec = feats["mean_rr"] / 1000.0
        feats["qtc_interval"] = float(feats["qt_interval"] / (rr_sec ** (1.0/3.0)))
    else:
        feats["qtc_interval"] = np.nan

    # =========================================================================
    # 7. Limb Leads Features (QRS axis, T-wave axis, QRS-T angle)
    # =========================================================================
    lead_I_qrs = get_avg_amp("I", ii_r_peaks)
    lead_aVF_qrs = get_avg_amp("aVF", ii_r_peaks)
    feats["qrs_axis"] = calculate_axis(lead_I_qrs, lead_aVF_qrs)
    
    t_lead_I_vals = []
    t_lead_aVF_vals = []
    for r_peak in ii_r_peaks:
        start = int(r_peak + 0.10 * fs)
        end = min(raw_signal.shape[0], int(r_peak + 0.40 * fs))
        if start < end:
            segment_I = clean_signals[start:end, LEADS_MAP["I"]] - lead_baselines["I"]
            segment_aVF = clean_signals[start:end, LEADS_MAP["aVF"]] - lead_baselines["aVF"]
            t_lead_I_vals.append(segment_I[np.argmax(np.abs(segment_I))])
            t_lead_aVF_vals.append(segment_aVF[np.argmax(np.abs(segment_aVF))])
            
    feats["t_wave_axis"] = calculate_axis(np.mean(t_lead_I_vals), np.mean(t_lead_aVF_vals)) if t_lead_I_vals else np.nan
    feats["qrs_t_angle"] = calculate_qrs_t_angle(feats["qrs_axis"], feats["t_wave_axis"])

    # =========================================================================
    # 8. Combined Ventricular-Voltage Feature
    # =========================================================================
    s_v1 = feats["v1_s_amplitude"]
    r_v5 = feats["v5_r_amplitude"]
    if not np.isnan(s_v1) and not np.isnan(r_v5):
        feats["sokolow_lyon"] = abs(s_v1) + r_v5
    else:
        feats["sokolow_lyon"] = np.nan

    # =========================================================================
    # Quality Control and Sanity Checks
    # =========================================================================
    qc_flags = []
    
    # Check Heart Rate
    if not np.isnan(feats["heart_rate"]):
        if feats["heart_rate"] <= 0 or feats["heart_rate"] > 350:
            qc_flags.append("failed_heart_rate")
            feats["heart_rate"] = np.nan
        elif feats["heart_rate"] < 40 or feats["heart_rate"] > 180:
            qc_flags.append("abnormal_heart_rate")
            
    # Check QRS Duration
    if not np.isnan(feats["qrs_duration"]):
        if feats["qrs_duration"] < 30 or feats["qrs_duration"] > 350:
            qc_flags.append("failed_qrs_duration")
            feats["qrs_duration"] = np.nan
        elif feats["qrs_duration"] < 60 or feats["qrs_duration"] > 140:
            qc_flags.append("abnormal_qrs_duration")
            
    # Check PR Interval
    if not np.isnan(feats["pr_interval"]):
        if feats["pr_interval"] < 50 or feats["pr_interval"] > 400:
            qc_flags.append("failed_pr_interval")
            feats["pr_interval"] = np.nan
        elif feats["pr_interval"] < 100 or feats["pr_interval"] > 240:
            qc_flags.append("abnormal_pr_interval")
            
    # Check QTc Interval
    if not np.isnan(feats["qtc_interval"]):
        if feats["qtc_interval"] < 150 or feats["qtc_interval"] > 900:
            qc_flags.append("failed_qtc_interval")
            feats["qtc_interval"] = np.nan
        elif feats["qtc_interval"] < 350 or feats["qtc_interval"] > 500:
            qc_flags.append("abnormal_qtc_interval")

    return feats, qc_failures + qc_flags

def main():
    parser = argparse.ArgumentParser(description="Extract custom ECG biomarkers.")
    parser.add_argument("--limit", type=int, default=ROW_LIMIT, help="Number of records to process")
    args = parser.parse_args()
    
    logger.info(f"Initializing dataset loader to process {args.limit} records...")
    
    loader = PTBXLLoader(
        root_dir=PTBXL_CONFIG["raw_dir"],
        database_csv=PTBXL_CONFIG["database_csv"],
        scp_csv=PTBXL_CONFIG["scp_csv"],
        resolution="hr"
    )
    
    metadata = loader.load_metadata()
    record_ids = metadata.index[:args.limit]
    
    logger.info(f"Extracting features for {len(record_ids)} records...")
    
    extracted_features = []
    qc_logs = []
    
    success_count = 0
    fail_count = 0
    qc_flag_count = 0
    
    for rid in tqdm(record_ids, desc="Extracting features"):
        try:
            record = loader.load_record(rid)
            feats, qc_issues = extract_record_features(record, SAMPLING_RATE)
            
            if feats is None:
                fail_count += 1
                qc_logs.append({"record_id": rid, "status": "failed", "issues": "; ".join(qc_issues)})
                continue
                
            # Parse targets
            classes = ["NORM", "MI", "STTC", "CD", "HYP"]
            label_str = str(record.metadata.get("diagnostic_classes", []))
            for c in classes:
                feats[c] = 1 if c in label_str else 0
                
            extracted_features.append(feats)
            success_count += 1
            
            if len(qc_issues) > 0:
                qc_flag_count += 1
                qc_logs.append({"record_id": rid, "status": "warning", "issues": "; ".join(qc_issues)})
            else:
                qc_logs.append({"record_id": rid, "status": "success", "issues": ""})
                
        except Exception as e:
            fail_count += 1
            qc_logs.append({"record_id": rid, "status": "failed", "issues": str(e)})

    # Create DataFrames
    df_features = pd.DataFrame(extracted_features)
    df_qc = pd.DataFrame(qc_logs)
    
    # Save CSVs
    os.makedirs(CSV_OUTPUT_PATH.parent, exist_ok=True)
    df_features.to_csv(CSV_OUTPUT_PATH, index=False)
    logger.info(f"Features saved to {CSV_OUTPUT_PATH}. Shape: {df_features.shape}")
    
    df_qc.to_csv(OUTPUT_DIR / "qc_logs.csv", index=False)
    
    # Missing percentages
    missing_pct = df_features.isna().mean() * 100.0
    
    # Write report
    with open(REPORT_OUTPUT_PATH, "w") as f:
        f.write("ECG BIOMARKER EXTRACTION SUMMARY REPORT\n")
        f.write("======================================\n\n")
        f.write(f"Number of records processed: {len(record_ids)}\n")
        f.write(f"Number of successful extractions: {success_count}\n")
        f.write(f"Number of failed extractions: {fail_count}\n")
        f.write(f"Number of records flagged for quality/warnings/sanity: {qc_flag_count}\n\n")
        f.write("Missing Value Percentage per Feature:\n")
        f.write("------------------------------------\n")
        for col, val in missing_pct.items():
            f.write(f"{col}: {val:.2f}%\n")
            
    logger.info(f"Report saved to {REPORT_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
