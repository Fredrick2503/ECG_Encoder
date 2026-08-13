from dataclasses import dataclass
from typing import Dict, List, Union
import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.stats
import scipy.signal
import warnings
import matplotlib.pyplot as plt
import logging

# Configure logger
logger = logging.getLogger("ECGFeatureExtractor")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

@dataclass
class ECGFeature:
    name: str
    value: float
    unit: str = ""
    description: str = ""

    def __repr__(self):
        if np.isnan(self.value):
            return f"{self.name}: NaN {self.unit}"
        return f"{self.name}: {self.value:.6f} {self.unit}"


class ECGFeatures:
    def __init__(self):
        self._features: Dict[str, ECGFeature] = {}
        self.debug_data: Dict[str, dict] = {}  # Map of lead_name -> debug_dict

    def add(self, name: str, value: float, unit: str = "", description: str = ""):
        self._features[name] = ECGFeature(
            name=name,
            value=float(value) if value is not None else np.nan,
            unit=unit,
            description=description
        )

    def get(self, name: str) -> ECGFeature:
        return self._features.get(name)

    def to_dict(self) -> Dict[str, float]:
        return {k: v.value for k, v in self._features.items()}

    def display(self):
        print("\n===== ECG FEATURES =====\n")
        for feature in self._features.values():
            print(feature)

    def __getitem__(self, key: str) -> ECGFeature:
        return self._features[key]

    def __iter__(self):
        return iter(self._features.values())

    def debug_intervals(self, lead_name: str = None):
        """Prints all beat-wise interval measurements for validation and debugging."""
        print(f"\n===== BEAT-WISE ECG INTERVALS DEBUG ({lead_name if lead_name else 'ALL LEADS'}) =====")
        
        leads_to_debug = [lead_name] if lead_name else list(self.debug_data.keys())
        for l_name in leads_to_debug:
            data = self.debug_data.get(l_name, {})
            if not data:
                continue
            print(f"\n--- Lead: {l_name} ---")
            pr = data.get("pr_intervals", [])
            qrs = data.get("qrs_durations", [])
            qt = data.get("qt_intervals", [])
            st = data.get("st_durations", [])
            tpe = data.get("tpe_intervals", [])
            r_onsets = data.get("r_onsets", [])
            r_offsets = data.get("r_offsets", [])
            
            num_beats = len(qrs)
            for i in range(num_beats):
                pr_val = pr[i] if i < len(pr) else np.nan
                qrs_val = qrs[i] if i < len(qrs) else np.nan
                qt_val = qt[i] if i < len(qt) else np.nan
                st_val = st[i] if i < len(st) else np.nan
                tpe_val = tpe[i] if i < len(tpe) else np.nan
                r_on = r_onsets[i] if i < len(r_onsets) else np.nan
                r_off = r_offsets[i] if i < len(r_offsets) else np.nan

                print(
                    f"Beat {i+1:02d}: "
                    f"PR={pr_val:.3f}s | "
                    f"QRS={qrs_val:.3f}s (Onset={r_on}, Offset={r_off}) | "
                    f"QT={qt_val:.3f}s | "
                    f"ST={st_val:.3f}s | "
                    f"Tp-e={tpe_val:.3f}s"
                )
        print("==========================================\n")


class ECGFeatureExtractor:
    STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    def __init__(self, fs: int = 500, leads: Union[str, List[str]] = "II"):
        self.fs = fs
        self.leads = leads

    def _integrate_signal(self, y, dx):
        if hasattr(np, "trapezoid"):
            return np.trapezoid(y, dx=dx)
        else:
            return np.trapz(y, dx=dx)

    def _get_amplitude(self, signal, idx):
        """Safely fetch signal amplitude by verifying index bounds and validity."""
        if idx is not None and not np.isnan(idx):
            int_idx = int(idx)
            if 0 <= int_idx < len(signal):
                return signal[int_idx]
        return np.nan

    def plot_delineation(self, ecg_signal: np.ndarray, save_path: str = None, lead_index: int = 1):
        """Debugging utility that plots the detected wave boundaries for a specific lead."""
        raw = np.asarray(ecg_signal)
        if raw.ndim == 1:
            lead_data = raw
        else:
            if lead_index < 0 or lead_index >= raw.shape[1]:
                import logging
                logging.getLogger("ECGFeatureExtractor").warning(
                    f"lead_index {lead_index} out of bounds for signal with {raw.shape[1]} leads. Falling back to index 0."
                )
                lead_index = 0
            lead_data = raw[:, lead_index]

        ecg_clean = nk.ecg_clean(lead_data, sampling_rate=self.fs)
        _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=self.fs)
        try:
            _, waves = nk.ecg_delineate(ecg_clean, rpeaks, sampling_rate=self.fs, method="dwt")
        except Exception as e:
            print(f"Delineation failed for plotting on lead {lead_index}:", e)
            return

        plt.figure(figsize=(12, 6))
        plt.plot(ecg_clean, label=f"ECG Cleaned (Lead Index {lead_index})", color="black", alpha=0.7)
        
        color_map = {
            "ECG_P_Onsets": "blue",
            "ECG_R_Onsets": "green",
            "ECG_R_Offsets": "red",
            "ECG_T_Peaks": "orange",
            "ECG_T_Offsets": "purple"
        }
        
        for key, color in color_map.items():
            indices = waves.get(key, [])
            valid_indices = [int(x) for x in indices if not np.isnan(x) and 0 <= int(x) < len(ecg_clean)]
            if valid_indices:
                plt.scatter(valid_indices, ecg_clean[valid_indices], label=key.replace("ECG_", ""), color=color, s=50, zorder=5)

        plt.title(f"ECG Delineation Fiducial Points (Lead Index {lead_index})")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude (mV)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Saved custom delineation plot to: {save_path}")
        plt.close()

    def _extract_single_lead(self, ecg_lead_signal: np.ndarray) -> Dict[str, float]:
        """Extracts all 47 original plus the new lead-specific biomarkers for a single lead."""
        res = {}
        
        # 1. Cleaning and Peak Detection
        ecg_clean = nk.ecg_clean(ecg_lead_signal, sampling_rate=self.fs)
        _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=self.fs)
        r_locs = rpeaks["ECG_R_Peaks"]
        num_beats = len(r_locs)

        if num_beats == 0:
            raise ValueError("No R peaks detected in the signal.")

        # 2. Delineation (DWT)
        try:
            _, waves = nk.ecg_delineate(ecg_clean, rpeaks, sampling_rate=self.fs, method="dwt")
        except Exception as e:
            logger.warning(f"Delineation failed: {str(e)}. Continuing with empty wave boundaries.")
            waves = {}

        # 3. HRV Analysis
        try:
            hrv = nk.hrv(rpeaks, sampling_rate=self.fs, show=False)
        except Exception as e:
            logger.warning(f"HRV computation failed: {str(e)}.")
            hrv = None

        def get_hrv_val(df, col):
            if df is not None and col in df.columns:
                val = df[col].iloc[0]
                return float(val) if not np.isnan(val) else np.nan
            return np.nan

        # RR Statistics & Heart Rate
        rr = np.diff(r_locs) / self.fs if len(r_locs) > 1 else np.array([])
        
        rr_mean = np.mean(rr) if len(rr) > 0 else np.nan
        rr_median = np.median(rr) if len(rr) > 0 else np.nan
        rr_min = np.min(rr) if len(rr) > 0 else np.nan
        rr_max = np.max(rr) if len(rr) > 0 else np.nan
        rr_range = rr_max - rr_min if len(rr) > 0 else np.nan
        rr_std = np.std(rr) if len(rr) > 0 else np.nan
        rr_var = np.var(rr) if len(rr) > 0 else np.nan
        rr_cv = rr_std / rr_mean if rr_mean > 0 else np.nan
        rr_iqr = np.percentile(rr, 75) - np.percentile(rr, 25) if len(rr) > 0 else np.nan
        
        try:
            rr_skew = scipy.stats.skew(rr) if len(rr) > 1 else np.nan
            rr_kurt = scipy.stats.kurtosis(rr) if len(rr) > 1 else np.nan
        except Exception:
            rr_skew = np.nan
            rr_kurt = np.nan

        hr = 60 / rr if len(rr) > 0 else np.array([])
        mean_hr = 60 / rr_mean if rr_mean > 0 else np.nan
        hr_std = np.std(hr) if len(hr) > 0 else np.nan
        min_hr = np.min(hr) if len(hr) > 0 else np.nan
        max_hr = np.max(hr) if len(hr) > 0 else np.nan

        # HRV (Time Domain)
        sdnn = get_hrv_val(hrv, "HRV_SDNN")
        rmssd = get_hrv_val(hrv, "HRV_RMSSD")
        sdsd = get_hrv_val(hrv, "HRV_SDSD")
        pnn50 = get_hrv_val(hrv, "HRV_pNN50")

        # HRV (Frequency Domain)
        signal_duration_sec = len(ecg_lead_signal) / self.fs
        if len(rr) < 100 or signal_duration_sec < 120.0:
            lf = np.nan
            hf = np.nan
            lf_hf = np.nan
        else:
            lf = get_hrv_val(hrv, "HRV_LF")
            hf = get_hrv_val(hrv, "HRV_HF")
            lf_hf = get_hrv_val(hrv, "HRV_LFHF")

        # HRV (Nonlinear)
        sd1 = get_hrv_val(hrv, "HRV_SD1")
        sd2 = get_hrv_val(hrv, "HRV_SD2")
        sd1_sd2 = get_hrv_val(hrv, "HRV_SD1SD2")
        
        samp_en = get_hrv_val(hrv, "HRV_SampEn")
        if np.isinf(samp_en) or np.isnan(samp_en):
            samp_en = np.nan

        # Wave boundary arrays - safely cast to Python lists
        def safe_get_list(key):
            val = waves.get(key, [np.nan] * num_beats)
            if val is None:
                return [np.nan] * num_beats
            if isinstance(val, (list, np.ndarray, pd.Series)):
                return list(val)
            return [val]

        p_onsets = safe_get_list("ECG_P_Onsets")
        p_peaks = safe_get_list("ECG_P_Peaks")
        p_offsets = safe_get_list("ECG_P_Offsets")
        q_peaks = safe_get_list("ECG_Q_Peaks")
        r_onsets = safe_get_list("ECG_R_Onsets")
        r_offsets = safe_get_list("ECG_R_Offsets")
        s_peaks = safe_get_list("ECG_S_Peaks")
        t_onsets = safe_get_list("ECG_T_Onsets")
        t_peaks = safe_get_list("ECG_T_Peaks")
        t_offsets = safe_get_list("ECG_T_Offsets")

        # Padding boundary arrays to match num_beats if needed
        for arr in [p_onsets, p_peaks, p_offsets, q_peaks, r_onsets, r_offsets, s_peaks, t_onsets, t_peaks, t_offsets]:
            if len(arr) < num_beats:
                arr.extend([np.nan] * (num_beats - len(arr)))

        pr_intervals = []
        qrs_durations = []
        qt_intervals = []
        st_durations = []
        tpe_intervals = []
        p_amps = []
        r_amps = []
        s_amps = []
        t_amps = []
        rs_ratios = []

        qrs_areas = []
        qrs_energies = []
        t_wave_areas = []
        st_slopes = []

        # New beat-wise features
        st_deviations = []
        st_elevations = []
        st_depressions = []
        t_widths = []
        t_symmetries = []
        t_slopes = []
        t_inversions = []
        t_biphasics = []
        p_durations = []
        p_areas = []
        p_symmetries = []
        q_durations = []
        q_depths = []
        pathological_qs = []
        fragmented_qrs = []
        notch_slurs = []

        # Newly added beat-wise biomarkers
        p_polarities = []
        qrs_amplitudes = []
        q_wave_amplitudes = []
        j_point_amplitudes = []
        st_segment_areas = []
        t_wave_polarities = []
        t_wave_peak_times = []
        r_prime_amplitudes = []
        s_prime_amplitudes = []
        st_t_relationships = []

        for i in range(num_beats):
            r_idx = r_locs[i]
            r_amp_val = self._get_amplitude(ecg_clean, r_idx)
            r_amps.append(r_amp_val)

            p_amp_val = self._get_amplitude(ecg_clean, p_peaks[i])
            p_amps.append(p_amp_val)

            s_amp_val = self._get_amplitude(ecg_clean, s_peaks[i])
            s_amps.append(s_amp_val)

            t_amp_val = self._get_amplitude(ecg_clean, t_peaks[i])
            t_amps.append(t_amp_val)

            if not np.isnan(r_amp_val) and not np.isnan(s_amp_val) and s_amp_val != 0:
                rs_ratios.append(r_amp_val / abs(s_amp_val))
            else:
                rs_ratios.append(np.nan)

            p_on = p_onsets[i]
            r_on = r_onsets[i]
            r_off = r_offsets[i]
            t_on = t_onsets[i]
            t_off = t_offsets[i]
            t_pk = t_peaks[i]
            p_off = p_offsets[i]
            p_pk = p_peaks[i]

            # Baseline at QRS onset
            baseline = self._get_amplitude(ecg_clean, r_on)
            if np.isnan(baseline):
                baseline = 0.0

            # PR Interval
            if not np.isnan(p_on) and not np.isnan(r_on):
                pr_intervals.append((r_on - p_on) / self.fs)
            else:
                pr_intervals.append(np.nan)

            # QRS Duration, Area, Energy
            if not np.isnan(r_on) and not np.isnan(r_off):
                qrs_durations.append((r_off - r_on) / self.fs)
                qrs_seg = ecg_clean[int(r_on):int(r_off)+1]
                if len(qrs_seg) > 0:
                    qrs_areas.append(self._integrate_signal(qrs_seg, dx=1/self.fs))
                    qrs_energies.append(np.sum(qrs_seg ** 2))
                else:
                    qrs_areas.append(np.nan)
                    qrs_energies.append(np.nan)
            else:
                qrs_durations.append(np.nan)
                qrs_areas.append(np.nan)
                qrs_energies.append(np.nan)

            # QT Interval
            if not np.isnan(r_on) and not np.isnan(t_off):
                qt_intervals.append((t_off - r_on) / self.fs)
            else:
                qt_intervals.append(np.nan)

            # ST Duration & Slope
            if not np.isnan(t_on) and not np.isnan(r_off):
                st_durations.append((t_on - r_off) / self.fs)
                st_dur_val = (t_on - r_off) / self.fs
                if st_dur_val > 0:
                    slope = (self._get_amplitude(ecg_clean, t_on) - self._get_amplitude(ecg_clean, r_off)) / st_dur_val
                    st_slopes.append(slope)
                else:
                    st_slopes.append(np.nan)
            else:
                st_durations.append(np.nan)
                st_slopes.append(np.nan)

            # Tp-e Interval
            if not np.isnan(t_pk) and not np.isnan(t_off):
                tpe_intervals.append((t_off - t_pk) / self.fs)
            else:
                tpe_intervals.append(np.nan)

            # T wave area
            if not np.isnan(t_on) and not np.isnan(t_off):
                t_seg = ecg_clean[int(t_on):int(t_off)+1]
                if len(t_seg) > 0:
                    t_wave_areas.append(self._integrate_signal(t_seg, dx=1/self.fs))
                else:
                    t_wave_areas.append(np.nan)
            else:
                t_wave_areas.append(np.nan)

            # --- NEW BIOMARKERS (Beat-wise) ---
            # ST deviation, elevation, depression
            if not np.isnan(r_off) and not np.isnan(t_on) and t_on > r_off:
                st_seg = ecg_clean[int(r_off):int(t_on)+1]
                if len(st_seg) > 0:
                    mean_st = np.mean(st_seg)
                    st_dev = mean_st - baseline
                    st_deviations.append(st_dev)
                    st_elevations.append(max(0.0, st_dev))
                    st_depressions.append(max(0.0, -st_dev))
                    st_segment_areas.append(self._integrate_signal(st_seg - baseline, dx=1/self.fs))
                else:
                    st_deviations.append(np.nan)
                    st_elevations.append(np.nan)
                    st_depressions.append(np.nan)
                    st_segment_areas.append(np.nan)
            else:
                st_deviations.append(np.nan)
                st_elevations.append(np.nan)
                st_depressions.append(np.nan)
                st_segment_areas.append(np.nan)

            # T Wave Morphology
            if not np.isnan(t_on) and not np.isnan(t_off) and t_off > t_on:
                t_widths.append((t_off - t_on) / self.fs)
                t_seg = ecg_clean[int(t_on):int(t_off)+1]
                
                # T-wave symmetry
                if not np.isnan(t_pk) and t_pk > t_on and t_off > t_pk:
                    t_symmetries.append((t_pk - t_on) / (t_off - t_pk + 1e-8))
                    
                    # T-wave slope
                    t_amp_v = self._get_amplitude(ecg_clean, t_pk)
                    t_on_amp = self._get_amplitude(ecg_clean, t_on)
                    t_off_amp = self._get_amplitude(ecg_clean, t_off)
                    upslope = (t_amp_v - t_on_amp) / ((t_pk - t_on) / self.fs) if t_pk > t_on else 0.0
                    downslope = (t_off_amp - t_amp_v) / ((t_off - t_pk) / self.fs) if t_off > t_pk else 0.0
                    t_slopes.append(max(abs(upslope), abs(downslope)))
                    t_wave_peak_times.append((t_pk - t_on) / self.fs)
                else:
                    t_symmetries.append(np.nan)
                    t_slopes.append(np.nan)
                    t_wave_peak_times.append(np.nan)

                # T inversion detection
                t_amp = self._get_amplitude(ecg_clean, t_pk) if not np.isnan(t_pk) else np.nan
                if not np.isnan(t_amp):
                    t_inversions.append(1.0 if (t_amp - baseline) < -0.05 else 0.0)
                    t_wave_polarities.append(np.sign(t_amp - baseline))
                else:
                    t_inversions.append(np.nan)
                    t_wave_polarities.append(np.nan)

                # Biphasic T-wave
                if len(t_seg) > 0:
                    t_seg_rel = t_seg - baseline
                    pos_pts = np.sum(t_seg_rel > 0.03)
                    neg_pts = np.sum(t_seg_rel < -0.03)
                    t_biphasics.append(1.0 if (pos_pts > 2 and neg_pts > 2) else 0.0)
                else:
                    t_biphasics.append(np.nan)
            else:
                t_widths.append(np.nan)
                t_symmetries.append(np.nan)
                t_slopes.append(np.nan)
                t_inversions.append(np.nan)
                t_biphasics.append(np.nan)
                t_wave_polarities.append(np.nan)
                t_wave_peak_times.append(np.nan)

            # P Wave Morphology
            if not np.isnan(p_on) and not np.isnan(p_off) and p_off > p_on:
                p_durations.append((p_off - p_on) / self.fs)
                p_seg = ecg_clean[int(p_on):int(p_off)+1]
                if len(p_seg) > 0:
                    p_areas.append(self._integrate_signal(p_seg, dx=1/self.fs))
                else:
                    p_areas.append(np.nan)

                if not np.isnan(p_pk) and p_pk > p_on and p_off > p_pk:
                    p_symmetries.append((p_pk - p_on) / (p_off - p_pk + 1e-8))
                else:
                    p_symmetries.append(np.nan)
            else:
                p_durations.append(np.nan)
                p_areas.append(np.nan)
                p_symmetries.append(np.nan)

            if not np.isnan(p_amp_val):
                p_polarities.append(np.sign(p_amp_val - baseline))
            else:
                p_polarities.append(np.nan)

            # QRS Morphology (Q-wave)
            q_val = np.nan
            q_dur = np.nan
            if not np.isnan(r_on) and not np.isnan(r_idx) and r_idx > r_on:
                qrs_start_seg = ecg_clean[int(r_on):int(r_idx)]
                if len(qrs_start_seg) > 0:
                    min_idx = np.argmin(qrs_start_seg)
                    q_pk_est = r_on + min_idx
                    q_val = ecg_clean[int(q_pk_est)]
                    q_dur = (r_idx - r_on) / (2 * self.fs)
                else:
                    q_val = 0.0
                    q_dur = 0.0
            
            q_depth = abs(q_val) if (q_val is not None and q_val < 0) else 0.0
            q_durations.append(q_dur if q_depth > 0 else 0.0)
            q_depths.append(q_depth)
            q_wave_amplitudes.append(q_depth)

            # Pathological Q-wave
            is_pathological = 0.0
            if q_depth > 0:
                if q_dur >= 0.04 or (not np.isnan(r_amp_val) and r_amp_val > 0 and q_depth >= 0.25 * r_amp_val):
                    is_pathological = 1.0
            pathological_qs.append(is_pathological)

            # Fragmented QRS
            if not np.isnan(r_on) and not np.isnan(r_off) and r_off > r_on:
                qrs_seg = ecg_clean[int(r_on):int(r_off)+1]
                if len(qrs_seg) > 3:
                    diffs = np.diff(qrs_seg)
                    extrema_count = np.sum(diffs[:-1] * diffs[1:] < 0)
                    fragmented_qrs.append(1.0 if extrema_count >= 4 else 0.0)
                else:
                    fragmented_qrs.append(0.0)
            else:
                fragmented_qrs.append(np.nan)

            # QRS Notching/Slurring
            if not np.isnan(r_on) and not np.isnan(r_off) and r_off > r_on:
                qrs_seg = ecg_clean[int(r_on):int(r_off)+1]
                if len(qrs_seg) > 3:
                    deriv = np.gradient(qrs_seg)
                    notch_count = np.sum(deriv[:-1] * deriv[1:] < 0)
                    slurred = np.any(np.abs(deriv) < 0.1 * np.max(np.abs(deriv)))
                    notch_slurs.append(1.0 if (notch_count >= 3 or slurred) else 0.0)
                else:
                    notch_slurs.append(0.0)
            else:
                notch_slurs.append(np.nan)

            # Newly added beat-wise biomarkers
            if not np.isnan(r_amp_val) and not np.isnan(s_amp_val):
                qrs_amplitudes.append(r_amp_val - s_amp_val)
            elif not np.isnan(r_amp_val):
                qrs_amplitudes.append(r_amp_val)
            else:
                qrs_amplitudes.append(np.nan)

            j_amp = self._get_amplitude(ecg_clean, r_off) if not np.isnan(r_off) else np.nan
            j_point_amplitudes.append(j_amp - baseline if not np.isnan(j_amp) else np.nan)

            # Secondary peaks (R', S')
            r_prime_val = 0.0
            s_prime_val = 0.0
            if not np.isnan(r_on) and not np.isnan(r_off) and r_off > r_on:
                qrs_seg = ecg_clean[int(r_on):int(r_off)+1]
                if len(qrs_seg) > 5:
                    peaks, _ = scipy.signal.find_peaks(qrs_seg)
                    valleys, _ = scipy.signal.find_peaks(-qrs_seg)
                    if len(peaks) > 1:
                        sorted_p = sorted([qrs_seg[p] for p in peaks])
                        r_prime_val = float(sorted_p[-2]) if len(sorted_p) >= 2 else 0.0
                    if len(valleys) > 1:
                        sorted_v = sorted([-qrs_seg[v] for v in valleys])
                        s_prime_val = float(sorted_v[-2]) if len(sorted_v) >= 2 else 0.0
            r_prime_amplitudes.append(r_prime_val)
            s_prime_amplitudes.append(s_prime_val)

            # ST-T relationship
            if len(st_slopes) > i and len(t_amps) > i and not np.isnan(st_slopes[i]) and not np.isnan(t_amps[i]):
                st_t_relationships.append(st_slopes[i] / (t_amps[i] + 1e-8))
            else:
                st_t_relationships.append(np.nan)

        # Store beat-wise debug measurements
        debug_dict = {
            "pr_intervals": pr_intervals,
            "qrs_durations": qrs_durations,
            "qt_intervals": qt_intervals,
            "st_durations": st_durations,
            "tpe_intervals": tpe_intervals,
            "r_onsets": r_onsets,
            "r_offsets": r_offsets
        }

        # Calculate averages for output
        mean_pr = np.nanmean(pr_intervals) if any(~np.isnan(pr_intervals)) else np.nan
        mean_qrs = np.nanmean(qrs_durations) if any(~np.isnan(qrs_durations)) else np.nan
        
        if mean_qrs > 0.160:
            warnings.warn(
                f"Mean QRS duration ({mean_qrs:.3f} s) exceeds 160 ms.",
                UserWarning
            )

        mean_qt = np.nanmean(qt_intervals) if any(~np.isnan(qt_intervals)) else np.nan
        mean_st_dur = np.nanmean(st_durations) if any(~np.isnan(st_durations)) else np.nan
        mean_tpe = np.nanmean(tpe_intervals) if any(~np.isnan(tpe_intervals)) else np.nan

        # QTc Bazett, Fridericia, Framingham, Hodges
        qtc_bazett = mean_qt / np.sqrt(rr_mean) if rr_mean > 0 else np.nan
        qtc_fridericia = mean_qt / np.cbrt(rr_mean) if rr_mean > 0 else np.nan
        qtc_framingham = mean_qt + 0.154 * (1.0 - rr_mean) if rr_mean > 0 else np.nan
        qtc_hodges = mean_qt + 0.00175 * (mean_hr - 60.0) if not np.isnan(mean_hr) else np.nan

        # Wave Amplitudes
        mean_p_amp = np.nanmean(p_amps) if any(~np.isnan(p_amps)) else np.nan
        mean_r_amp = np.nanmean(r_amps) if any(~np.isnan(r_amps)) else np.nan
        mean_s_amp = np.nanmean(s_amps) if any(~np.isnan(s_amps)) else np.nan
        mean_t_amp = np.nanmean(t_amps) if any(~np.isnan(t_amps)) else np.nan
        mean_rs_ratio = np.nanmean(rs_ratios) if any(~np.isnan(rs_ratios)) else np.nan

        # Morphology
        mean_qrs_area = np.nanmean(qrs_areas) if any(~np.isnan(qrs_areas)) else np.nan
        mean_qrs_energy = np.nanmean(qrs_energies) if any(~np.isnan(qrs_energies)) else np.nan
        mean_t_area = np.nanmean(t_wave_areas) if any(~np.isnan(t_wave_areas)) else np.nan
        mean_st_slope = np.nanmean(st_slopes) if any(~np.isnan(st_slopes)) else np.nan

        # QT Variability
        qt_clean = np.asarray(qt_intervals)
        qt_clean = qt_clean[~np.isnan(qt_clean)]
        qt_var = np.var(qt_clean) if len(qt_clean) > 0 else np.nan

        # QT Dispersion
        qt_disp = (np.nanmax(qt_intervals) - np.nanmin(qt_intervals)) if len(qt_clean) > 0 else np.nan
        if qt_disp > 0.120:
            warnings.warn(
                f"QT Dispersion ({qt_disp:.3f} s) exceeds 120 ms.",
                UserWarning
            )

        tpe_qt_ratio = mean_tpe / mean_qt if not np.isnan(mean_tpe) and not np.isnan(mean_qt) and mean_qt != 0 else np.nan

        # RR-QT correlation & covariance
        rr_arr = np.asarray(rr)
        valid_len = min(len(rr_arr), len(qt_intervals) - 1)
        if valid_len > 1:
            rr_aligned = rr_arr[:valid_len]
            qt_aligned = np.asarray(qt_intervals[1:valid_len+1])
            mask = ~np.isnan(rr_aligned) & ~np.isnan(qt_aligned)
            if np.sum(mask) > 1:
                rr_clean = rr_aligned[mask]
                qt_clean = qt_aligned[mask]
                rr_qt_corr = np.corrcoef(rr_clean, qt_clean)[0, 1]
                rr_qt_cov = np.cov(rr_clean, qt_clean)[0, 1]
            else:
                rr_qt_corr = np.nan
                rr_qt_cov = np.nan
        else:
            rr_qt_corr = np.nan
            rr_qt_cov = np.nan

        # Averages for new features
        mean_st_dev = np.nanmean(st_deviations) if any(~np.isnan(st_deviations)) else np.nan
        mean_st_elev = np.nanmean(st_elevations) if any(~np.isnan(st_elevations)) else np.nan
        mean_st_depr = np.nanmean(st_depressions) if any(~np.isnan(st_depressions)) else np.nan
        max_st_elev = np.nanmax(st_elevations) if any(~np.isnan(st_elevations)) else np.nan
        max_st_depr = np.nanmax(st_depressions) if any(~np.isnan(st_depressions)) else np.nan
        mean_t_width = np.nanmean(t_widths) if any(~np.isnan(t_widths)) else np.nan
        mean_t_sym = np.nanmean(t_symmetries) if any(~np.isnan(t_symmetries)) else np.nan
        mean_t_slope = np.nanmean(t_slopes) if any(~np.isnan(t_slopes)) else np.nan
        mean_t_inversion = np.nanmean(t_inversions) if any(~np.isnan(t_inversions)) else np.nan
        mean_t_biphasic = np.nanmean(t_biphasics) if any(~np.isnan(t_biphasics)) else np.nan
        mean_p_dur = np.nanmean(p_durations) if any(~np.isnan(p_durations)) else np.nan
        mean_p_area = np.nanmean(p_areas) if any(~np.isnan(p_areas)) else np.nan
        mean_p_sym = np.nanmean(p_symmetries) if any(~np.isnan(p_symmetries)) else np.nan
        mean_q_dur = np.nanmean(q_durations) if any(~np.isnan(q_durations)) else np.nan
        mean_q_depth = np.nanmean(q_depths) if any(~np.isnan(q_depths)) else np.nan
        mean_path_q = np.nanmean(pathological_qs) if any(~np.isnan(pathological_qs)) else np.nan
        mean_frag_qrs = np.nanmean(fragmented_qrs) if any(~np.isnan(fragmented_qrs)) else np.nan
        mean_notch_slur = np.nanmean(notch_slurs) if any(~np.isnan(notch_slurs)) else np.nan

        # Newly averaged biomarkers
        mean_p_polarity = np.nanmean(p_polarities) if any(~np.isnan(p_polarities)) else np.nan
        mean_qrs_amplitude = np.nanmean(qrs_amplitudes) if any(~np.isnan(qrs_amplitudes)) else np.nan
        mean_q_wave_amplitude = np.nanmean(q_wave_amplitudes) if any(~np.isnan(q_wave_amplitudes)) else np.nan
        mean_j_point_amplitude = np.nanmean(j_point_amplitudes) if any(~np.isnan(j_point_amplitudes)) else np.nan
        mean_st_segment_area = np.nanmean(st_segment_areas) if any(~np.isnan(st_segment_areas)) else np.nan
        mean_t_wave_polarity = np.nanmean(t_wave_polarities) if any(~np.isnan(t_wave_polarities)) else np.nan
        mean_t_wave_peak_time = np.nanmean(t_wave_peak_times) if any(~np.isnan(t_wave_peak_times)) else np.nan
        mean_r_prime_amplitude = np.nanmean(r_prime_amplitudes) if any(~np.isnan(r_prime_amplitudes)) else np.nan
        mean_s_prime_amplitude = np.nanmean(s_prime_amplitudes) if any(~np.isnan(s_prime_amplitudes)) else np.nan
        mean_st_t_relationship = np.nanmean(st_t_relationships) if any(~np.isnan(st_t_relationships)) else np.nan
        nn50 = np.sum(np.abs(np.diff(rr)) > 0.05) if len(rr) > 1 else 0.0

        # Populate output results
        res["RR_Mean"] = rr_mean
        res["RR_Median"] = rr_median
        res["RR_Min"] = rr_min
        res["RR_Max"] = rr_max
        res["RR_Range"] = rr_range
        res["RR_STD"] = rr_std
        res["RR_Variance"] = rr_var
        res["RR_CV"] = rr_cv
        res["RR_IQR"] = rr_iqr
        res["RR_Skewness"] = rr_skew
        res["RR_Kurtosis"] = rr_kurt
        
        res["Mean_HR"] = mean_hr
        res["HR_STD"] = hr_std
        res["Min_HR"] = min_hr
        res["Max_HR"] = max_hr

        res["SDNN"] = sdnn
        res["RMSSD"] = rmssd
        res["SDSD"] = sdsd
        res["pNN50"] = pnn50
        res["NN50"] = nn50

        res["LF_Power"] = lf
        res["HF_Power"] = hf
        res["LF_HF_Ratio"] = lf_hf

        res["SD1"] = sd1
        res["SD2"] = sd2
        res["SD1_SD2_Ratio"] = sd1_sd2
        res["Sample_Entropy"] = samp_en

        res["PR_Interval"] = mean_pr
        res["QRS_Duration"] = mean_qrs
        res["QT_Interval"] = mean_qt
        res["QTc_Bazett"] = qtc_bazett
        res["QTc_Fridericia"] = qtc_fridericia
        res["QTc_Framingham"] = qtc_framingham
        res["QTc_Hodges"] = qtc_hodges
        res["ST_Duration"] = mean_st_dur

        res["P_Amplitude"] = mean_p_amp
        res["R_Amplitude"] = mean_r_amp
        res["S_Amplitude"] = mean_s_amp
        res["T_Amplitude"] = mean_t_amp
        res["R_S_Ratio"] = mean_rs_ratio

        res["QRS_Area"] = mean_qrs_area
        res["QRS_Energy"] = mean_qrs_energy
        res["T_wave_Area"] = mean_t_area
        res["ST_Slope"] = mean_st_slope

        res["QT_Variability"] = qt_var
        res["QT_Dispersion"] = qt_disp
        res["Tp_e_Interval"] = mean_tpe
        res["Tp_e_QT_Ratio"] = tpe_qt_ratio
        res["RR_QT_Correlation"] = rr_qt_corr
        res["RR_QT_Covariance"] = rr_qt_cov

        # New clinically important biomarkers (lead-specific)
        res["ST_Deviation"] = mean_st_dev
        res["ST_Elevation"] = mean_st_elev
        res["ST_Depression"] = mean_st_depr
        res["Max_ST_Elevation"] = max_st_elev
        res["Max_ST_Depression"] = max_st_depr
        res["T_wave_Width"] = mean_t_width
        res["T_wave_Symmetry"] = mean_t_sym
        res["T_wave_Slope"] = mean_t_slope
        res["T_wave_Inversion"] = mean_t_inversion
        res["Biphasic_T_wave"] = mean_t_biphasic
        res["P_wave_Duration"] = mean_p_dur
        res["P_wave_Area"] = mean_p_area
        res["P_wave_Symmetry"] = mean_p_sym
        res["Q_wave_Duration"] = mean_q_dur
        res["Q_wave_Depth"] = mean_q_depth
        res["Pathological_Q_wave"] = mean_path_q
        res["Fragmented_QRS"] = mean_frag_qrs
        res["QRS_Notching_Slurring"] = mean_notch_slur

        # Newly added biomarkers populated
        res["P_wave_Polarity"] = mean_p_polarity
        res["QRS_Amplitude"] = mean_qrs_amplitude
        res["Q_wave_Amplitude"] = mean_q_wave_amplitude
        res["J_point_Amplitude"] = mean_j_point_amplitude
        res["ST_Segment_Area"] = mean_st_segment_area
        res["T_wave_Polarity"] = mean_t_wave_polarity
        res["T_wave_Peak_Time"] = mean_t_wave_peak_time
        res["R_prime_Amplitude"] = mean_r_prime_amplitude
        res["S_prime_Amplitude"] = mean_s_prime_amplitude
        res["ST_T_Relationship"] = mean_st_t_relationship

        return res, debug_dict

    def extract(self, ecg_signal: np.ndarray, leads: Union[str, List[str]] = None) -> ECGFeatures:
        """
        Extracts biomarkers from the configured leads of the ECG signal.
        Supports Lead II only (backward compatible), selected leads, or all 12 leads.
        """
        if leads is None:
            leads = self.leads

        # Resolve lead list
        if isinstance(leads, str):
            if leads.lower() == "all":
                requested_leads = self.STANDARD_12_LEADS
            else:
                requested_leads = [leads]
        else:
            requested_leads = list(leads)

        features = ECGFeatures()
        
        # Handle 1D vs 2D shapes
        raw_signal = np.asarray(ecg_signal, dtype=np.float32)
        is_1d = (raw_signal.ndim == 1)

        # Map lead names to columns
        lead_to_col = {}
        if is_1d or raw_signal.shape[1] == 1:
            # Single channel input: map the first requested lead to column 0
            lead_to_col[requested_leads[0]] = 0
        else:
            # Multi-channel input
            for idx, name in enumerate(self.STANDARD_12_LEADS):
                if idx < raw_signal.shape[1]:
                    lead_to_col[name] = idx

        lead_results = {}
        
        # Determine if we should prefix features with lead names
        # To maintain exact backward compatibility:
        # If we are only extracting "II" and it's a single-lead extraction, do not prefix
        prefix_lead_names = True
        if len(requested_leads) == 1 and requested_leads[0] == "II":
            prefix_lead_names = False

        # Extract features lead-by-lead with robust handling
        from joblib import Parallel, delayed

        def run_single_lead(name):
            col_idx = lead_to_col.get(name)
            if col_idx is None:
                return name, None, None
            lead_data = raw_signal if is_1d else raw_signal[:, col_idx]
            try:
                single_res, single_debug = self._extract_single_lead(lead_data)
                return name, single_res, single_debug
            except Exception as e:
                logger.error(f"Failed to extract features for Lead {name}: {str(e)}")
                return name, {}, {}

        parallel_results = Parallel(n_jobs=-1)(
            delayed(run_single_lead)(name) for name in requested_leads
        )

        for name, single_res, single_debug in parallel_results:
            if single_res is not None:
                lead_results[name] = single_res
                features.debug_data[name] = single_debug

        # Add lead-specific features to the container
        for lead_name, res in lead_results.items():
            prefix = f"lead_{lead_name}_" if prefix_lead_names else ""
            
            # If the extraction failed completely for a lead, populate NaN for all expected features
            if not res:
                all_keys = [
                    "RR_Mean", "RR_Median", "RR_Min", "RR_Max", "RR_Range", "RR_STD", "RR_Variance", "RR_CV", "RR_IQR", "RR_Skewness", "RR_Kurtosis",
                    "Mean_HR", "HR_STD", "Min_HR", "Max_HR", "SDNN", "RMSSD", "SDSD", "pNN50", "LF_Power", "HF_Power", "LF_HF_Ratio",
                    "SD1", "SD2", "SD1_SD2_Ratio", "Sample_Entropy", "PR_Interval", "QRS_Duration", "QT_Interval", "QTc_Bazett", "QTc_Fridericia", "ST_Duration",
                    "P_Amplitude", "R_Amplitude", "S_Amplitude", "T_Amplitude", "R_S_Ratio", "QRS_Area", "QRS_Energy", "T_wave_Area", "ST_Slope",
                    "QT_Variability", "QT_Dispersion", "Tp_e_Interval", "Tp_e_QT_Ratio", "RR_QT_Correlation", "RR_QT_Covariance",
                    "ST_Deviation", "ST_Elevation", "ST_Depression", "Max_ST_Elevation", "Max_ST_Depression", "T_wave_Width", "T_wave_Symmetry", "T_wave_Slope",
                    "T_wave_Inversion", "Biphasic_T_wave", "P_wave_Duration", "P_wave_Area", "P_wave_Symmetry", "Q_wave_Duration", "Q_wave_Depth",
                    "Pathological_Q_wave", "Fragmented_QRS", "QRS_Notching_Slurring"
                ]
                res = {k: np.nan for k in all_keys}

            for k, val in res.items():
                features.add(f"{prefix}{k}", val)

        # --- MULTI-LEAD MORPHOLOGY & GLOBAL BIOMARKERS ---
        st_elev_count = 0
        st_depr_count = 0
        p_wave_durations = []

        for lead_name, res in lead_results.items():
            if res:
                st_elev = res.get("ST_Elevation", np.nan)
                st_depr = res.get("ST_Depression", np.nan)
                p_dur = res.get("P_wave_Duration", np.nan)

                if not np.isnan(st_elev) and st_elev > 0.05:
                    st_elev_count += 1
                if not np.isnan(st_depr) and st_depr > 0.05:
                    st_depr_count += 1
                if not np.isnan(p_dur):
                    p_wave_durations.append(p_dur)

        # Add global multi-lead features if multiple leads were processed
        if len(requested_leads) > 1:
            features.add("Num_Leads_ST_Elevation", st_elev_count, "", "Number of leads with significant ST elevation")
            features.add("Num_Leads_ST_Depression", st_depr_count, "", "Number of leads with significant ST depression")
            
            p_disp = (np.max(p_wave_durations) - np.min(p_wave_durations)) if len(p_wave_durations) > 1 else np.nan
            features.add("P_Wave_Dispersion", p_disp, "sec", "Difference between max and min P-wave duration across leads")

        # P terminal force (Lead V1)
        if "V1" in requested_leads and "V1" in features.debug_data:
            col_v1 = lead_to_col.get("V1")
            if col_v1 is not None and not is_1d:
                v1_sig = raw_signal[:, col_v1]
                v1_clean = nk.ecg_clean(v1_sig, sampling_rate=self.fs)
                
                _, v1_rpeaks = nk.ecg_peaks(v1_clean, sampling_rate=self.fs)
                try:
                    _, v1_waves = nk.ecg_delineate(v1_clean, v1_rpeaks, sampling_rate=self.fs, method="dwt")
                    p_onsets_v1 = v1_waves.get("ECG_P_Onsets", [])
                    p_offsets_v1 = v1_waves.get("ECG_P_Offsets", [])
                    r_onsets_v1 = v1_waves.get("ECG_R_Onsets", [])
                    
                    forces = []
                    for i in range(len(v1_rpeaks["ECG_R_Peaks"])):
                        if i < len(p_onsets_v1) and i < len(p_offsets_v1):
                            p_on = p_onsets_v1[i]
                            p_off = p_offsets_v1[i]
                            r_on = r_onsets_v1[i] if i < len(r_onsets_v1) else np.nan
                            
                            baseline = self._get_amplitude(v1_clean, r_on) if not np.isnan(r_on) else 0.0
                            if np.isnan(baseline):
                                baseline = 0.0

                            if not np.isnan(p_on) and not np.isnan(p_off) and p_off > p_on:
                                p_seg = v1_clean[int(p_on):int(p_off)+1]
                                if len(p_seg) > 0:
                                    min_idx = np.argmin(p_seg)
                                    nadir_val = p_seg[min_idx]
                                    nadir_amp = nadir_val - baseline
                                    
                                    if nadir_amp < 0:
                                        term_dur = (len(p_seg) - min_idx) / self.fs
                                        forces.append(term_dur * abs(nadir_amp))
                    
                    mean_force = np.mean(forces) if forces else 0.0
                    features.add("P_Terminal_Force_V1", mean_force, "mV*sec", "P-wave terminal force in lead V1")
                except Exception:
                    features.add("P_Terminal_Force_V1", np.nan, "mV*sec", "P-wave terminal force in lead V1")
            else:
                features.add("P_Terminal_Force_V1", np.nan, "mV*sec", "P-wave terminal force in lead V1")

        # R-wave progression (V1-V6)
        precordial_leads = ["V1", "V2", "V3", "V4", "V5", "V6"]
        if all(lead in requested_leads for lead in precordial_leads):
            r_amps = []
            for lead in precordial_leads:
                if lead in lead_results and "R_Amplitude" in lead_results[lead]:
                    r_amps.append(lead_results[lead]["R_Amplitude"])
                else:
                    r_amps.append(np.nan)
            
            v3_r_amp = r_amps[2]
            features.add("R_Wave_Progression_V3", v3_r_amp, "mV", "R-wave amplitude in V3")
            if not np.isnan(v3_r_amp):
                poor_prog = 1.0 if v3_r_amp <= 0.3 else 0.0
                features.add("Poor_R_Wave_Progression", poor_prog, "", "Poor R-wave progression flag")
            else:
                features.add("Poor_R_Wave_Progression", np.nan, "", "Poor R-wave progression flag")

        # Electrical Axis (P, QRS, T axes) in frontal plane
        if "I" in requested_leads and "aVF" in requested_leads:
            res_I = lead_results.get("I", {})
            res_aVF = lead_results.get("aVF", {})
            
            if res_I and res_aVF:
                net_I = res_I.get("R_Amplitude", 0.0) - abs(res_I.get("S_Amplitude", 0.0))
                net_aVF = res_aVF.get("R_Amplitude", 0.0) - abs(res_aVF.get("S_Amplitude", 0.0))
                qrs_axis = np.degrees(np.arctan2(net_aVF, net_I))
                features.add("QRS_Axis", qrs_axis, "degrees", "Frontal plane QRS electrical axis")

                p_I = res_I.get("P_Amplitude", 0.0)
                p_aVF = res_aVF.get("P_Amplitude", 0.0)
                p_axis = np.degrees(np.arctan2(p_aVF, p_I))
                features.add("P_Axis", p_axis, "degrees", "Frontal plane P electrical axis")

                t_I = res_I.get("T_Amplitude", 0.0)
                t_aVF = res_aVF.get("T_Amplitude", 0.0)
                t_axis = np.degrees(np.arctan2(t_aVF, t_I))
                features.add("T_Axis", t_axis, "degrees", "Frontal plane T electrical axis")

                diff_axis = abs(qrs_axis - t_axis)
                qrs_t_angle = diff_axis if diff_axis <= 180 else 360.0 - diff_axis
                features.add("QRS_T_Angle", qrs_t_angle, "degrees", "Angle between QRS and T electrical axes")

        # Multi-lead global clinical features
        sokolow = np.nan
        cornell = np.nan
        cornell_product = np.nan
        
        res_v1 = lead_results.get("V1", {})
        res_v5 = lead_results.get("V5", {})
        res_v6 = lead_results.get("V6", {})
        s_v1 = abs(res_v1.get("S_Amplitude", np.nan)) if res_v1 else np.nan
        r_v5 = abs(res_v5.get("R_Amplitude", np.nan)) if res_v5 else np.nan
        r_v6 = abs(res_v6.get("R_Amplitude", np.nan)) if res_v6 else np.nan
        if not np.isnan(s_v1) and (not np.isnan(r_v5) or not np.isnan(r_v6)):
            sokolow = s_v1 + max(np.nan_to_num(r_v5), np.nan_to_num(r_v6))
        features.add("Sokolow_Lyon_Voltage", sokolow, "mV")

        res_avl = lead_results.get("aVL", {})
        res_v3 = lead_results.get("V3", {})
        r_avl = abs(res_avl.get("R_Amplitude", np.nan)) if res_avl else np.nan
        s_v3 = abs(res_v3.get("S_Amplitude", np.nan)) if res_v3 else np.nan
        if not np.isnan(r_avl) and not np.isnan(s_v3):
            cornell = r_avl + s_v3
        features.add("Cornell_Voltage", cornell, "mV")

        qrs_dur_avl = res_avl.get("QRS_Duration", np.nan) if res_avl else np.nan
        if not np.isnan(cornell) and not np.isnan(qrs_dur_avl):
            cornell_product = cornell * qrs_dur_avl * 1000.0
        features.add("Cornell_Voltage_Duration_Product", cornell_product, "mV*ms")

        # QRS Voltage Dispersion across leads
        qrs_amps = []
        for l_name in requested_leads:
            res_lead = lead_results.get(l_name, {})
            if res_lead:
                r_a = res_lead.get("R_Amplitude", np.nan)
                s_a = res_lead.get("S_Amplitude", np.nan)
                if not np.isnan(r_a) and not np.isnan(s_a):
                    qrs_amps.append(r_a + abs(s_a))
        qrs_disp = np.std(qrs_amps) if len(qrs_amps) > 1 else np.nan
        features.add("QRS_Voltage_Dispersion", qrs_disp, "mV")

        return features
