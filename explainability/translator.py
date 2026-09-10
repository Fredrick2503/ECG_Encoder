import numpy as np
import neurokit2 as nk
import pandas as pd
from typing import Dict, Tuple, List

class GradCAMTranslator:
    """
    Translates Grad-CAM spatial/spectrogram attributions back to the original 
    time-domain ECG signal and cross-references with clinical ECG landmarks.
    """
    def __init__(self, sampling_rate: int = 100, n_fft: int = 64, hop_length: int = 32):
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_duration = n_fft / sampling_rate

    def map_spectrogram_to_time(self, stft_time_index: int, total_samples: int) -> Tuple[int, int]:
        """
        Maps an STFT time index (column) back to the original signal sample interval.
        The STFT window at index t_stft is centered at t_stft * hop_length.
        """
        center_sample = stft_time_index * self.hop_length
        start_sample = max(0, center_sample - self.n_fft // 2)
        end_sample = min(total_samples, center_sample + self.n_fft // 2)
        return int(start_sample), int(end_sample)

    def find_highest_attribution_region(self, gradcam_attr: np.ndarray, total_samples: int) -> Dict[int, dict]:
        """
        Finds the time region with the highest attribution for each lead.
        gradcam_attr: (leads, freq_bins, time_steps) or (1, freq_bins, time_steps)
        """
        regions = {}
        # If gradcam is 1-channel (e.g. from 2D CNN), we analyze it identically for all leads
        if gradcam_attr.ndim == 2:
            gradcam_attr = np.expand_dims(gradcam_attr, axis=0)
        
        if gradcam_attr.shape[0] == 1:
            gradcam_attr = np.repeat(gradcam_attr, 12, axis=0)
            
        num_leads = gradcam_attr.shape[0]
        
        for lead_idx in range(num_leads):
            # Sum attribution across frequency bins to get time-wise attribution
            time_attr = np.sum(gradcam_attr[lead_idx], axis=0)
            best_stft_idx = np.argmax(time_attr)
            max_attr_val = time_attr[best_stft_idx]
            
            start_sample, end_sample = self.map_spectrogram_to_time(best_stft_idx, total_samples)
            
            # Calculate total and peak attribution for ranking
            total_lead_attr = float(np.sum(gradcam_attr[lead_idx]))
            peak_lead_attr = float(np.max(gradcam_attr[lead_idx]))
            
            regions[lead_idx] = {
                'stft_index': best_stft_idx,
                'start_sample': start_sample,
                'end_sample': end_sample,
                'attribution_strength': float(max_attr_val),
                'total_attribution': total_lead_attr,
                'peak_attribution': peak_lead_attr
            }
            
        return regions

    def rank_leads(self, regions: Dict[int, dict]) -> List[Tuple[int, float, float]]:
        """
        Ranks the leads by their total attribution strength.
        Returns a sorted list of tuples: (lead_index, total_attribution, peak_attribution)
        """
        ranking = []
        for lead_idx, data in regions.items():
            ranking.append((lead_idx, data.get('total_attribution', 0.0), data.get('peak_attribution', 0.0)))
        # Sort by total attribution descending
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def delineate_ecg(self, signal: np.ndarray) -> dict:
        """
        Uses neurokit2 to delineate the ECG signal (P, QRS, T waves).
        Expects a 1D signal array for a single lead.
        """
        try:
            # Clean signal and find R-peaks
            cleaned = nk.ecg_clean(signal, sampling_rate=self.sampling_rate)
            _, info = nk.ecg_peaks(cleaned, sampling_rate=self.sampling_rate)
            rpeaks = info['ECG_R_Peaks']
            
            if len(rpeaks) == 0:
                return {}
            
            # Delineate using DWT
            _, waves_peak = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=self.sampling_rate, method="dwt")
            return waves_peak
        except Exception as e:
            # Fallback if NeuroKit2 fails
            return {}

    def analyze_attribution_overlap(self, signal: np.ndarray, regions: Dict[int, dict]) -> Dict[int, dict]:
        """
        Cross-references the high-attribution regions with delineated ECG waves.
        signal: (leads, time_steps)
        """
        num_leads = signal.shape[0]
        results = {}
        
        for lead_idx in range(num_leads):
            region = regions.get(lead_idx)
            if not region:
                continue
                
            start_s = region['start_sample']
            end_s = region['end_sample']
            
            waves = self.delineate_ecg(signal[lead_idx])
            overlapping_waves = []
            
            # Helper to check overlap
            def is_in_region(idx_list):
                if idx_list is None:
                    return False
                if hasattr(idx_list, 'tolist'):
                    vals = idx_list.tolist()
                elif isinstance(idx_list, (list, np.ndarray)):
                    vals = list(idx_list)
                else:
                    vals = [idx_list]
                for val in vals:
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        if start_s <= val <= end_s:
                            return True
                return False
                
            if is_in_region(waves.get('ECG_P_Peaks')):
                overlapping_waves.append('P-Wave')
            if is_in_region(waves.get('ECG_R_Peaks')) or is_in_region(waves.get('ECG_Q_Peaks')) or is_in_region(waves.get('ECG_S_Peaks')):
                overlapping_waves.append('QRS-Complex')
            if is_in_region(waves.get('ECG_T_Peaks')):
                overlapping_waves.append('T-Wave')
                
            # If QRS not in region, check if it's an ST segment (between S/R and T)
            if 'QRS-Complex' not in overlapping_waves and 'T-Wave' not in overlapping_waves:
                r_peaks = waves.get('ECG_R_Peaks', [])
                t_peaks = waves.get('ECG_T_Peaks', [])
                
                # Convert to lists if they are numpy/pandas
                if hasattr(r_peaks, 'tolist'): r_peaks = r_peaks.tolist()
                if hasattr(t_peaks, 'tolist'): t_peaks = t_peaks.tolist()
                
                if isinstance(r_peaks, list) and isinstance(t_peaks, list):
                    # Filter out NaNs
                    r_peaks_clean = [r for r in r_peaks if r is not None and not np.isnan(r)]
                    t_peaks_clean = [t for t in t_peaks if t is not None and not np.isnan(t)]
                    for r in r_peaks_clean:
                        for t in t_peaks_clean:
                            if r < t:
                                # ST segment roughly between R and T
                                st_center = (r + t) / 2
                                if start_s <= st_center <= end_s:
                                    overlapping_waves.append('ST-Segment')
                                    break
                        if 'ST-Segment' in overlapping_waves:
                            break
                            
            if not overlapping_waves:
                overlapping_waves.append('None/Unidentified')
                
            results[lead_idx] = {
                **region,
                'overlapping_waves': overlapping_waves
            }
            
        return results
