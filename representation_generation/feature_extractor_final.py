"""
PTB-XL Biomarker Feature Extractor
===================================

Standalone ECG biomarker extractor.

Run from project root:

    python representation_generation/feature_extractor_final.py --limit 100

Full dataset:

    python representation_generation/feature_extractor_final.py

Output:

    data/processed/biomarker_features.csv

The output contains one flat row per ECG and individual numeric columns.

Biomarkers:
    RR_Mean
    QRS_Duration
    PR_Interval
    QT_Interval
    QTc_Bazett
    ST_Duration
    P_wave_Duration
    R_Amplitude
    P_Amplitude
    T_Amplitude
    ST_Deviation
    Q_Amplitude
    R_S_Ratio
    QRS_Energy
    SDNN
    RMSSD
    pNN50
    pNN20
    SDRR_RMSSD_Ratio
    HRV_Triangular_Index
    LF_Power
    HF_Power
    LF_HF_Ratio
    Total_Power
    Sample_Entropy
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from scipy import signal

# NeuroKit is used only for robust R-peak detection.
# Morphology is extracted using our own waveform-based method.
try:
    import neurokit2 as nk
except ImportError:
    nk = None


# ============================================================
# CONFIGURATION
# ============================================================

LEADS = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

FEATURE_COLUMNS = [
    "RR_Mean",
    "QRS_Duration",
    "PR_Interval",
    "QT_Interval",
    "QTc_Bazett",
    "ST_Duration",
    "P_wave_Duration",
    "R_Amplitude",
    "P_Amplitude",
    "T_Amplitude",
    "ST_Deviation",
    "Q_Amplitude",
    "R_S_Ratio",
    "QRS_Energy",
    "SDNN",
    "RMSSD",
    "pNN50",
    "pNN20",
    "SDRR_RMSSD_Ratio",
    "HRV_Triangular_Index",
    "LF_Power",
    "HF_Power",
    "LF_HF_Ratio",
    "Total_Power",
    "Sample_Entropy",
]

OUTPUT_COLUMNS = [
    "ecg_id",
    "patient_id",
    "age",
    "sex",
] + FEATURE_COLUMNS


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - BiomarkerExtraction - %(levelname)s - %(message)s",
)

logger = logging.getLogger("BiomarkerExtraction")

warnings.filterwarnings(
    "ignore",
    message="Too few peaks detected",
)

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
)


# ============================================================
# PATHS
# ============================================================

def find_ptbxl_root() -> Path:
    """
    Locate PTB-XL automatically.

    Expected:
        project/
            data/
                raw/
                    ptbxl/
                        ptbxl_database.csv

    Also supports:
        data/ptbxl/
    """

    project_root = Path(__file__).resolve().parents[1]

    candidates = [
        project_root / "data" / "raw" / "ptbxl",
        project_root / "data" / "ptbxl",
        project_root / "data" / "raw" / "PTB-XL",
        project_root / "data" / "PTB-XL",
    ]

    for path in candidates:
        if (path / "ptbxl_database.csv").exists():
            return path

    # Recursive fallback
    for csv_file in project_root.rglob("ptbxl_database.csv"):
        return csv_file.parent

    raise FileNotFoundError(
        "Could not find ptbxl_database.csv.\n"
        "Expected PTB-XL at:\n"
        "  data/raw/ptbxl/\n"
        "or\n"
        "  data/ptbxl/"
    )


# ============================================================
# BASIC SIGNAL PROCESSING
# ============================================================

def bandpass_filter(
    x: np.ndarray,
    fs: float,
    low: float = 0.5,
    high: float = 40.0,
) -> np.ndarray:
    """
    0.5-40 Hz Butterworth bandpass.
    """

    x = np.asarray(x, dtype=float)

    if len(x) < 100:
        return x.copy()

    nyquist = fs / 2.0

    high = min(high, nyquist * 0.90)

    if high <= low:
        return x.copy()

    b, a = signal.butter(
        3,
        [low / nyquist, high / nyquist],
        btype="band",
    )

    try:
        return signal.filtfilt(b, a, x)
    except Exception:
        return x.copy()


def robust_baseline(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Remove slow baseline drift.
    """

    window = int(round(fs * 0.2))

    if window < 3:
        return x - np.median(x)

    if window % 2 == 0:
        window += 1

    if window >= len(x):
        return x - np.median(x)

    baseline = signal.medfilt(x, kernel_size=window)

    return x - baseline


def preprocess(x: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if not np.isfinite(x).all():
        good = np.isfinite(x)

        if not np.any(good):
            return np.zeros_like(x)

        x = np.interp(
            np.arange(len(x)),
            np.flatnonzero(good),
            x[good],
        )

    x = bandpass_filter(x, fs)
    x = robust_baseline(x, fs)

    return x


# ============================================================
# R PEAK DETECTION
# ============================================================

def detect_r_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Robust R-peak detection.

    NeuroKit is preferred.
    scipy fallback is used if NeuroKit fails.
    """

    x = np.asarray(x, dtype=float)

    # --------------------------------------------------------
    # NeuroKit
    # --------------------------------------------------------

    if nk is not None:

        try:
            _, info = nk.ecg_peaks(
                x,
                sampling_rate=fs,
                correct_artifacts=False,
            )

            peaks = np.asarray(
                info.get("ECG_R_Peaks", []),
                dtype=int,
            )

            if len(peaks) >= 2:
                return peaks

        except Exception:
            pass

    # --------------------------------------------------------
    # scipy fallback
    # --------------------------------------------------------

    distance = max(1, int(0.25 * fs))

    prominence = max(
        np.std(x) * 0.25,
        1e-6,
    )

    peaks, _ = signal.find_peaks(
        x,
        distance=distance,
        prominence=prominence,
    )

    if len(peaks) >= 2:
        return peaks

    # Try inverted signal.
    peaks, _ = signal.find_peaks(
        -x,
        distance=distance,
        prominence=prominence,
    )

    return peaks.astype(int)


# ============================================================
# RR / HRV
# ============================================================

def extract_rr_features(
    r_peaks: np.ndarray,
    fs: float,
) -> dict:

    result = {
        "RR_Mean": np.nan,
        "SDNN": np.nan,
        "RMSSD": np.nan,
        "pNN50": np.nan,
        "pNN20": np.nan,
        "SDRR_RMSSD_Ratio": np.nan,
        "HRV_Triangular_Index": np.nan,
    }

    if len(r_peaks) < 2:
        return result

    rr = np.diff(r_peaks) / fs

    # Remove physiologically implausible intervals.
    rr = rr[
        (rr >= 0.25) &
        (rr <= 2.5)
    ]

    if len(rr) == 0:
        return result

    result["RR_Mean"] = float(np.mean(rr))

    if len(rr) >= 2:
        sdnn = np.std(rr, ddof=1)
        diff_rr = np.diff(rr)

        rmssd = np.sqrt(
            np.mean(diff_rr ** 2)
        )

        result["SDNN"] = float(sdnn)
        result["RMSSD"] = float(rmssd)

        result["pNN50"] = float(
            100.0 * np.mean(
                np.abs(diff_rr) > 0.050
            )
        )

        result["pNN20"] = float(
            100.0 * np.mean(
                np.abs(diff_rr) > 0.020
            )
        )

        if rmssd > 1e-12:
            result["SDRR_RMSSD_Ratio"] = float(
                sdnn / rmssd
            )

    # HRV triangular index.
    if len(rr) >= 3:

        try:
            bins = max(
                5,
                int(np.sqrt(len(rr)) * 3)
            )

            hist, _ = np.histogram(
                rr,
                bins=bins,
            )

            max_bin = np.max(hist)

            if max_bin > 0:
                result["HRV_Triangular_Index"] = float(
                    len(rr) / max_bin
                )

        except Exception:
            pass

    return result


# ============================================================
# SAMPLE ENTROPY
# ============================================================

def sample_entropy(
    values: np.ndarray,
    m: int = 2,
    r_ratio: float = 0.2,
) -> float:

    x = np.asarray(values, dtype=float)

    if len(x) < 10:
        return np.nan

    x = x[np.isfinite(x)]

    if len(x) < 10:
        return np.nan

    sd = np.std(x)

    if sd < 1e-12:
        return 0.0

    r = r_ratio * sd

    def count_matches(mm: int) -> int:

        templates = np.array([
            x[i:i + mm]
            for i in range(len(x) - mm + 1)
        ])

        count = 0

        for i in range(len(templates)):
            distances = np.max(
                np.abs(
                    templates[i + 1:] -
                    templates[i]
                ),
                axis=1,
            )

            count += np.sum(
                distances <= r
            )

        return int(count)

    try:

        a = count_matches(m + 1)
        b = count_matches(m)

        if a == 0 or b == 0:
            return np.nan

        return float(
            -np.log(a / b)
        )

    except Exception:
        return np.nan


# ============================================================
# FREQUENCY DOMAIN
# ============================================================

def extract_frequency_features(
    rr: np.ndarray,
) -> dict:
    """
    Frequency-domain HRV.

    PTB-XL is generally a short recording.
    Therefore these values are only calculated when there
    are enough RR samples for a meaningful estimate.

    We do NOT fabricate values.
    """

    result = {
        "LF_Power": np.nan,
        "HF_Power": np.nan,
        "LF_HF_Ratio": np.nan,
        "Total_Power": np.nan,
    }

    if len(rr) < 20:
        return result

    # RR series needs enough duration for useful spectral analysis.
    total_duration = np.sum(rr)

    if total_duration < 60:
        return result

    try:

        times = np.cumsum(rr)

        times = times - times[0]

        if times[-1] <= 0:
            return result

        # Interpolate at 4 Hz.
        new_times = np.arange(
            0,
            times[-1],
            0.25,
        )

        if len(new_times) < 32:
            return result

        rr_interp = np.interp(
            new_times,
            times,
            rr,
        )

        rr_interp = signal.detrend(
            rr_interp
        )

        freqs, power = signal.welch(
            rr_interp,
            fs=4.0,
            nperseg=min(
                256,
                len(rr_interp)
            ),
        )

        lf_mask = (
            (freqs >= 0.04) &
            (freqs < 0.15)
        )

        hf_mask = (
            (freqs >= 0.15) &
            (freqs <= 0.40)
        )

        total_mask = (
            (freqs >= 0.0033) &
            (freqs <= 0.40)
        )

        lf = np.trapezoid(
            power[lf_mask],
            freqs[lf_mask],
        ) if np.any(lf_mask) else np.nan

        hf = np.trapezoid(
            power[hf_mask],
            freqs[hf_mask],
        ) if np.any(hf_mask) else np.nan

        total = np.trapezoid(
            power[total_mask],
            freqs[total_mask],
        ) if np.any(total_mask) else np.nan

        result["LF_Power"] = float(lf)
        result["HF_Power"] = float(hf)
        result["Total_Power"] = float(total)

        if (
            np.isfinite(lf) and
            np.isfinite(hf) and
            hf > 1e-12
        ):
            result["LF_HF_Ratio"] = float(
                lf / hf
            )

    except Exception:
        pass

    return result


# ============================================================
# LOCAL MORPHOLOGY
# ============================================================

def local_extreme(
    x: np.ndarray,
    center: int,
    left: int,
    right: int,
    mode: str,
):
    start = max(0, center + left)
    end = min(len(x), center + right)

    if end <= start + 2:
        return None

    segment = x[start:end]

    if mode == "max":
        idx = np.argmax(segment)
    else:
        idx = np.argmin(segment)

    return start + int(idx)


def median_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.median(values))


def extract_morphology(
    x: np.ndarray,
    r_peaks: np.ndarray,
    fs: float,
) -> dict:

    result = {
        "QRS_Duration": np.nan,
        "PR_Interval": np.nan,
        "QT_Interval": np.nan,
        "QTc_Bazett": np.nan,
        "ST_Duration": np.nan,
        "P_wave_Duration": np.nan,
        "R_Amplitude": np.nan,
        "P_Amplitude": np.nan,
        "T_Amplitude": np.nan,
        "ST_Deviation": np.nan,
        "Q_Amplitude": np.nan,
        "R_S_Ratio": np.nan,
        "QRS_Energy": np.nan,
    }

    if len(r_peaks) == 0:
        return result

    qrs_durations = []
    pr_intervals = []
    qt_intervals = []
    st_durations = []
    p_durations = []

    r_values = []
    p_values = []
    t_values = []
    st_values = []
    q_values = []
    rs_ratios = []
    qrs_energies = []

    beat_rrs = []

    for i, r in enumerate(r_peaks):

        if i > 0:
            rr = (r - r_peaks[i - 1]) / fs

            if 0.25 <= rr <= 2.5:
                beat_rrs.append(rr)

        # ----------------------------------------------------
        # Q
        # Search 80 ms before R.
        # ----------------------------------------------------

        q = local_extreme(
            x,
            r,
            int(-0.10 * fs),
            int(-0.015 * fs),
            "min",
        )

        # ----------------------------------------------------
        # S
        # Search 20-120 ms after R.
        # ----------------------------------------------------

        s = local_extreme(
            x,
            r,
            int(0.015 * fs),
            int(0.12 * fs),
            "min",
        )

        # ----------------------------------------------------
        # T
        # Search 100-450 ms after R.
        # ----------------------------------------------------

        t = local_extreme(
            x,
            r,
            int(0.10 * fs),
            int(0.45 * fs),
            "max",
        )

        # If T is predominantly negative, try minimum.
        if t is None:
            t = local_extreme(
                x,
                r,
                int(0.10 * fs),
                int(0.45 * fs),
                "min",
            )

        # ----------------------------------------------------
        # P
        # Search 250-40 ms before R.
        # ----------------------------------------------------

        p = local_extreme(
            x,
            r,
            int(-0.30 * fs),
            int(-0.04 * fs),
            "max",
        )

        if p is None:
            p = local_extreme(
                x,
                r,
                int(-0.30 * fs),
                int(-0.04 * fs),
                "min",
            )

        # ----------------------------------------------------
        # Amplitudes
        # ----------------------------------------------------

        r_amp = x[r]

        if np.isfinite(r_amp):
            r_values.append(r_amp)

        if p is not None:
            p_amp = x[p]

            if np.isfinite(p_amp):
                p_values.append(p_amp)

        if t is not None:
            t_amp = x[t]

            if np.isfinite(t_amp):
                t_values.append(t_amp)

        if q is not None:
            q_amp = x[q]

            if np.isfinite(q_amp):
                q_values.append(q_amp)

        # ----------------------------------------------------
        # QRS duration
        # ----------------------------------------------------

        if q is not None and s is not None:

            duration = (s - q) / fs

            if 0.04 <= duration <= 0.20:
                qrs_durations.append(duration)

        # ----------------------------------------------------
        # PR interval
        # ----------------------------------------------------

        if p is not None and q is not None:

            duration = (q - p) / fs

            if 0.08 <= duration <= 0.30:
                pr_intervals.append(duration)

        # ----------------------------------------------------
        # QT interval
        # ----------------------------------------------------

        if q is not None and t is not None:

            duration = (t - q) / fs

            if 0.20 <= duration <= 0.65:
                qt_intervals.append(duration)

        # ----------------------------------------------------
        # ST duration
        # ----------------------------------------------------

        if s is not None and t is not None:

            duration = (t - s) / fs

            if 0.03 <= duration <= 0.30:
                st_durations.append(duration)

        # ----------------------------------------------------
        # P wave duration
        #
        # Approximate local P-wave width.
        # ----------------------------------------------------

        if p is not None:

            p_amp = x[p]

            baseline_left = max(
                0,
                p - int(0.10 * fs)
            )

            baseline_right = min(
                len(x),
                p + int(0.10 * fs)
            )

            baseline = np.median(
                np.concatenate([
                    x[baseline_left:p]
                    if p > baseline_left
                    else np.array([]),

                    x[p:baseline_right]
                    if baseline_right > p
                    else np.array([]),
                ])
            )

            threshold = (
                abs(p_amp - baseline) * 0.20
            )

            if threshold > 1e-6:

                start = p

                while (
                    start > 0 and
                    abs(x[start] - baseline) > threshold
                ):
                    start -= 1

                end = p

                while (
                    end < len(x) - 1 and
                    abs(x[end] - baseline) > threshold
                ):
                    end += 1

                duration = (end - start) / fs

                if 0.04 <= duration <= 0.20:
                    p_durations.append(duration)

        # ----------------------------------------------------
        # ST deviation
        #
        # J point ≈ S.
        # Measure ~80 ms after S.
        # ----------------------------------------------------

        if s is not None:

            st_index = min(
                len(x) - 1,
                s + int(0.08 * fs)
            )

            baseline_start = max(
                0,
                r - int(0.20 * fs)
            )

            baseline_end = max(
                baseline_start + 1,
                r - int(0.10 * fs)
            )

            baseline = np.median(
                x[baseline_start:baseline_end]
            )

            st_value = x[st_index] - baseline

            if np.isfinite(st_value):
                st_values.append(st_value)

        # ----------------------------------------------------
        # R/S ratio
        # ----------------------------------------------------

        if s is not None:

            s_depth = abs(x[s])

            if s_depth > 1e-6:

                ratio = abs(
                    x[r] / s_depth
                )

                if np.isfinite(ratio) and ratio < 1000:
                    rs_ratios.append(ratio)

        # ----------------------------------------------------
        # QRS energy
        # ----------------------------------------------------

        if q is not None and s is not None:

            start = min(q, s)
            end = max(q, s)

            if end > start:

                qrs = x[start:end + 1]

                energy = np.sqrt(
                    np.mean(qrs ** 2)
                )

                if np.isfinite(energy):
                    qrs_energies.append(energy)

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    result["QRS_Duration"] = median_or_nan(
        qrs_durations
    )

    result["PR_Interval"] = median_or_nan(
        pr_intervals
    )

    result["QT_Interval"] = median_or_nan(
        qt_intervals
    )

    result["ST_Duration"] = median_or_nan(
        st_durations
    )

    result["P_wave_Duration"] = median_or_nan(
        p_durations
    )

    result["R_Amplitude"] = median_or_nan(
        r_values
    )

    result["P_Amplitude"] = median_or_nan(
        p_values
    )

    result["T_Amplitude"] = median_or_nan(
        t_values
    )

    result["ST_Deviation"] = median_or_nan(
        st_values
    )

    result["Q_Amplitude"] = median_or_nan(
        q_values
    )

    result["R_S_Ratio"] = median_or_nan(
        rs_ratios
    )

    result["QRS_Energy"] = median_or_nan(
        qrs_energies
    )

    # --------------------------------------------------------
    # QTc Bazett
    # --------------------------------------------------------

    qt = result["QT_Interval"]

    if np.isfinite(qt):

        if len(beat_rrs) > 0:
            rr = np.median(beat_rrs)
        else:
            rr = np.nan

        if np.isfinite(rr) and rr > 0:
            result["QTc_Bazett"] = float(
                qt / np.sqrt(rr)
            )

    return result


# ============================================================
# SINGLE ECG
# ============================================================

def extract_ecg(
    record_path: Path,
    fs: float,
) -> dict:

    signal_data, fields = wfdb.rdsamp(
        str(record_path)
    )

    signal_data = np.asarray(
        signal_data,
        dtype=float,
    )

    # --------------------------------------------------------
    # Ensure 12 leads
    # --------------------------------------------------------

    available_names = fields.get(
        "sig_name",
        []
    )

    lead_indices = []

    for lead in LEADS:

        if lead in available_names:
            lead_indices.append(
                available_names.index(lead)
            )

    if len(lead_indices) == 0:

        # PTB-XL should already be in standard order.
        n = min(
            signal_data.shape[1],
            12,
        )

        lead_indices = list(range(n))

    # --------------------------------------------------------
    # Process all leads
    # --------------------------------------------------------

    lead_results = []

    for idx in lead_indices:

        x = signal_data[:, idx]

        x = preprocess(
            x,
            fs,
        )

        r_peaks = detect_r_peaks(
            x,
            fs,
        )

        if len(r_peaks) < 2:
            continue

        rr_features = extract_rr_features(
            r_peaks,
            fs,
        )

        morphology = extract_morphology(
            x,
            r_peaks,
            fs,
        )

        # RR array for frequency / entropy.
        rr = np.diff(r_peaks) / fs

        rr = rr[
            (rr >= 0.25) &
            (rr <= 2.5)
        ]

        frequency = extract_frequency_features(
            rr
        )

        entropy = sample_entropy(
            rr
        )

        result = {}

        result.update(
            rr_features
        )

        result.update(
            morphology
        )

        result.update(
            frequency
        )

        result["Sample_Entropy"] = entropy

        lead_results.append(
            result
        )

    if len(lead_results) == 0:

        return {
            feature: np.nan
            for feature in FEATURE_COLUMNS
        }

    # --------------------------------------------------------
    # Combine 12 leads.
    #
    # Median is used instead of selecting one arbitrary lead.
    # --------------------------------------------------------

    combined = {}

    for feature in FEATURE_COLUMNS:

        values = []

        for result in lead_results:

            value = result.get(
                feature,
                np.nan
            )

            if np.isfinite(value):
                values.append(value)

        combined[feature] = (
            float(np.median(values))
            if values
            else np.nan
        )

    return combined


# ============================================================
# MAIN EXTRACTION
# ============================================================

import os

def process_single_row(args):
    ecg_id, filename, ptbxl_root, patient_id, age, sex = args
    try:
        record_path = ptbxl_root / str(filename)
        _, fields = wfdb.rdsamp(str(record_path), sampto=1)
        fs = float(fields["fs"])
        features = extract_ecg(record_path, fs)
        output_row = {
            "ecg_id": ecg_id,
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
        }
        output_row.update(features)
        return output_row, None
    except Exception as exc:
        return None, (ecg_id, str(exc))

def main():

    parser = argparse.ArgumentParser(
        description="Extract ECG biomarkers from PTB-XL."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of ECG records to process. Omit for full dataset.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Find PTB-XL
    # --------------------------------------------------------

    ptbxl_root = find_ptbxl_root()

    logger.info(
        "PTB-XL root: %s",
        ptbxl_root,
    )

    metadata_path = (
        ptbxl_root /
        "ptbxl_database.csv"
    )

    metadata = pd.read_csv(
        metadata_path,
        index_col=0,
    )

    # --------------------------------------------------------
    # Determine sampling frequency.
    #
    # PTB-XL has both 100 Hz and 500 Hz records.
    # We use the record's actual fs.
    # --------------------------------------------------------

    if args.limit is not None:

        metadata = metadata.head(
            args.limit
        )

    logger.info(
        "Records selected: %d",
        len(metadata),
    )

    logger.info(
        "Extracting 12 leads; model-facing features: %d",
        len(FEATURE_COLUMNS),
    )

    rows = []
    failed = 0

    # --------------------------------------------------------
    # Process records in parallel
    # --------------------------------------------------------
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    logger.info("Preparing metadata parameters...")
    tasks_args = []
    for ecg_id, row in metadata.iterrows():
        filename = row.get("filename_hr", None)
        if pd.isna(filename):
            filename = row.get("filename_lr", None)
        if pd.isna(filename):
            continue
        patient_id = row.get("patient_id", np.nan)
        age = row.get("age", np.nan)
        sex = row.get("sex", np.nan)
        tasks_args.append((ecg_id, filename, ptbxl_root, patient_id, age, sex))

    num_workers = max(1, (os.cpu_count() or 2) - 1)
    logger.info(f"Using {num_workers} processes for parallel feature extraction...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_row, arg): arg[0] for arg in tasks_args}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting biomarkers"):
            ecg_id = futures[future]
            try:
                row_data, err = future.result()
                if err is not None:
                    failed += 1
                    logger.warning("Failed ECG %s: %s", err[0], err[1])
                else:
                    rows.append(row_data)
            except Exception as exc:
                failed += 1
                logger.warning("Failed ECG %s: %s", ecg_id, exc)

    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )

    # --------------------------------------------------------
    # Ensure numeric feature columns
    # --------------------------------------------------------

    for column in FEATURE_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Save ONE CSV
    # --------------------------------------------------------

    project_root = Path(
        __file__
    ).resolve().parents[1]

    output_dir = (
        project_root /
        "data" /
        "processed"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir /
        "biomarker_features.csv"
    )

    df.to_csv(
        output_file,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    logger.info("")
    logger.info(
        "========================================"
    )

    logger.info(
        "BIOMARKER EXTRACTION COMPLETE"
    )

    logger.info(
        "========================================"
    )

    logger.info(
        "Successful records : %d",
        len(df),
    )

    logger.info(
        "Failed records     : %d",
        failed,
    )

    logger.info(
        "Feature columns     : %d",
        len(FEATURE_COLUMNS),
    )

    logger.info(
        "Output CSV          : %s",
        output_file,
    )

    logger.info(
        "CSV shape           : %s",
        df.shape,
    )

    logger.info(
        "========================================"
    )

    # --------------------------------------------------------
    # Missingness
    # --------------------------------------------------------

    if len(df) > 0:

        missing = (
            df[FEATURE_COLUMNS]
            .isna()
            .mean()
            .sort_values(
                ascending=False
            )
        )

        logger.info("")
        logger.info(
            "Feature missingness:"
        )

        logger.info(
            "\n%s",
            missing.to_string()
        )

        # ----------------------------------------------------
        # Important warning only
        # ----------------------------------------------------

        high_missing = missing[
            missing > 0.20
        ]

        if len(high_missing) > 0:

            logger.warning(
                "Features with >20%% missing values:"
            )

            logger.warning(
                "\n%s",
                high_missing.to_string()
            )

        logger.info("")
        logger.info(
            "First 5 rows:"
        )

        logger.info(
            "\n%s",
            df.head().to_string()
        )


if __name__ == "__main__":
    main()