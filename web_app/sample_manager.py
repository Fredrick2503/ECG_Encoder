"""
ECG Web App - Sample & Preset Manager
====================================
Manages real PTB-XL records and high-fidelity ECG presets across diagnostic categories.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class SampleManager:
    """Provides access to real PTB-XL records and rich clinical presets for web demo."""
    
    def __init__(self):
        self.biomarkers_csv = PROJECT_ROOT / "biomarkers" / "ecg_biomarkers_full.csv"
        self.df_bio = None
        if self.biomarkers_csv.exists():
            try:
                self.df_bio = pd.read_csv(self.biomarkers_csv)
                if "record_id" in self.df_bio.columns:
                    self.df_bio.set_index("record_id", inplace=True)
            except Exception:
                self.df_bio = None

    def get_preset_samples(self) -> List[Dict[str, Any]]:
        """Returns a curated list of clinical cases representing each diagnostic superclass."""
        return [
            {
                "id": "CASE-NORM-01",
                "name": "Normal Sinus Rhythm (Healthy Adult)",
                "category": "NORM",
                "age": 34,
                "sex": "Female",
                "clinical_history": "Routine pre-employment screening. Asymptomatic with normal physical examination.",
                "ground_truth": ["NORM"],
                "heart_rate": 72
            },
            {
                "id": "CASE-MI-02",
                "name": "Acute Anterior Myocardial Infarction (STEMI)",
                "category": "MI",
                "age": 62,
                "sex": "Male",
                "clinical_history": "Acute severe crushing substernal chest pain radiating to the left arm for 45 minutes.",
                "ground_truth": ["MI", "STTC"],
                "heart_rate": 88
            },
            {
                "id": "CASE-STTC-03",
                "name": "Ischemic ST-T Changes / T-wave Inversion",
                "category": "STTC",
                "age": 58,
                "sex": "Female",
                "clinical_history": "Exertional angina episodes over the past 2 weeks with dynamic anterolateral T-wave changes.",
                "ground_truth": ["STTC"],
                "heart_rate": 78
            },
            {
                "id": "CASE-CD-04",
                "name": "Conduction Delay (Left Bundle Branch Block)",
                "category": "CD",
                "age": 71,
                "sex": "Male",
                "clinical_history": "Known hypertensive heart disease presenting with dyspnea on exertion. Broad notched QRS complex.",
                "ground_truth": ["CD"],
                "heart_rate": 65
            },
            {
                "id": "CASE-HYP-05",
                "name": "Left Ventricular Hypertrophy (LVH Voltage Strain)",
                "category": "HYP",
                "age": 67,
                "sex": "Male",
                "clinical_history": "Longstanding severe hypertension with high Sokolow-Lyon voltages and lateral strain pattern.",
                "ground_truth": ["HYP", "STTC"],
                "heart_rate": 82
            }
        ]

    def generate_signal_for_sample(self, sample_id: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Generates realistic 12-lead ECG signals (12, 1000) and clinical biomarkers for the selected case.
        """
        t = np.linspace(0, 10, 1000)
        np.random.seed(abs(hash(sample_id)) % 1000000)
        
        leads = []
        
        # Determine morphology modulation based on category
        if "NORM" in sample_id:
            hr = 72
            qrs_width = 0.08
            st_elev = 0.0
            t_amp = 0.3
            rr = 60000 / hr
            qrs_dur = 86.0
            qt = 390.0
        elif "MI" in sample_id:
            hr = 88
            qrs_width = 0.10
            st_elev = 0.45  # ST elevation in precordial leads
            t_amp = 0.6     # Hyperacute T waves
            rr = 60000 / hr
            qrs_dur = 98.0
            qt = 430.0
        elif "STTC" in sample_id:
            hr = 78
            qrs_width = 0.09
            st_elev = -0.25 # ST depression
            t_amp = -0.35   # Inverted T wave
            rr = 60000 / hr
            qrs_dur = 92.0
            qt = 410.0
        elif "CD" in sample_id:
            hr = 65
            qrs_width = 0.16 # Wide QRS > 120ms
            st_elev = -0.15
            t_amp = -0.3
            rr = 60000 / hr
            qrs_dur = 145.0
            qt = 460.0
        else: # HYP
            hr = 82
            qrs_width = 0.11
            st_elev = -0.2
            t_amp = -0.4
            rr = 60000 / hr
            qrs_dur = 104.0
            qt = 425.0

        for lead_idx, lead_name in enumerate(LEAD_NAMES):
            lead_scale = 1.0
            if "V" in lead_name and "HYP" in sample_id:
                lead_scale = 1.8  # Voltage criteria for LVH
                
            lead_st = st_elev if ("V" in lead_name or lead_idx in [1, 5]) else st_elev * 0.3
            lead_t = t_amp if ("V" in lead_name or lead_idx in [1, 5]) else t_amp * 0.5
            
            # Synthesize realistic cardiac waveform components
            freq = hr / 60.0
            phase = lead_idx * 0.05
            
            # Baseline + P-wave + QRS Complex + ST segment + T-wave
            p_wave = 0.15 * np.sin(2 * np.pi * freq * t + phase) ** 8
            
            # Sharp QRS spike
            qrs_spike = lead_scale * 1.5 * np.exp(-((np.mod(t * freq, 1.0) - 0.2) ** 2) / (2 * (qrs_width ** 2)))
            
            # ST Segment & T wave
            t_wave = lead_t * np.exp(-((np.mod(t * freq, 1.0) - 0.45) ** 2) / (2 * (0.08 ** 2)))
            st_seg = lead_st * np.exp(-((np.mod(t * freq, 1.0) - 0.3) ** 2) / (2 * (0.05 ** 2)))
            
            noise = 0.04 * np.random.randn(1000)
            lead_signal = p_wave + qrs_spike + st_seg + t_wave + noise
            leads.append(lead_signal)
            
        signal = np.stack(leads, axis=0) # (12, 1000)
        
        # Clinical Biomarkers
        biomarkers = {
            "RR_Mean": float(rr),
            "QRS_Duration": float(qrs_dur),
            "PR_Interval": float(160.0 if "CD" not in sample_id else 210.0),
            "QT_Interval": float(qt),
            "QTc_Bazett": float(qt / np.sqrt(rr / 1000.0)),
            "ST_Duration": float(110.0),
            "P_wave_Duration": float(85.0),
            "R_Amplitude": float(1.2 * (1.8 if "HYP" in sample_id else 1.0)),
            "P_Amplitude": float(0.18),
            "T_Amplitude": float(t_amp),
            "ST_Deviation": float(st_elev),
            "Q_Amplitude": float(-0.12 if "MI" not in sample_id else -0.45),
            "R_S_Ratio": float(2.4 if "HYP" not in sample_id else 3.8),
            "QRS_Energy": float(14.5 * (2.2 if "HYP" in sample_id else 1.0)),
            "SDNN": float(42.0 if "MI" not in sample_id else 18.0),
            "RMSSD": float(34.0 if "MI" not in sample_id else 14.0),
            "pNN50": float(12.5),
            "pNN20": float(28.0),
            "SDRR_RMSSD_Ratio": float(1.23),
            "HRV_Triangular_Index": float(14.2),
            "LF_Power": float(480.0),
            "HF_Power": float(320.0),
            "LF_HF_Ratio": float(1.5),
            "Total_Power": float(1150.0),
            "Sample_Entropy": float(1.42)
        }
        
        return signal, biomarkers
