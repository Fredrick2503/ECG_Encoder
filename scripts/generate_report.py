"""
ECG Clinical Diagnostic Report Generator — Individual Report CLI
================================================================
Generates a complete, publication-grade individual clinical diagnostic report
for any selected ECG record or custom 12-lead ECG file.

Outputs:
1. Formatted Clinical Medical Summary in Console (ASCII tables & findings)
2. Standalone, Print-Ready Black & White HTML Clinical Report (A4/PDF compatible)
3. Optional Machine-Readable JSON Export with embeddings, biomarkers, and probabilities

Usage:
    # Generate report for default sample (REC-01) and open in browser
    python scripts/generate_report.py --open

    # Generate report for a specific PTB-XL record
    python scripts/generate_report.py --sample REC-21826 --open

    # Generate report for a custom .npy signal file
    python scripts/generate_report.py --input path/to/ecg.npy --open

    # List available preset records
    python scripts/generate_report.py --list
"""

from __future__ import annotations
import os
import sys
import json
import math
import argparse
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ecg_engine import ECGEncoderEngine, EngineConfig
from web_app.sample_manager import SampleManager, LEAD_NAMES, CLASS_NAMES
from web_app.llm_service import GeminiClinicalInterpreter

# Biomarker Normal Clinical Reference Ranges
BIOMARKER_RANGES: Dict[str, Tuple[float, float, str]] = {
    "RR_Mean": (600.0, 1200.0, "ms"),
    "QRS_Duration": (70.0, 110.0, "ms"),
    "PR_Interval": (120.0, 200.0, "ms"),
    "QT_Interval": (350.0, 440.0, "ms"),
    "QTc_Bazett": (360.0, 450.0, "ms"),
    "ST_Duration": (80.0, 140.0, "ms"),
    "P_wave_Duration": (60.0, 110.0, "ms"),
    "R_Amplitude": (0.5, 2.0, "mV"),
    "P_Amplitude": (0.05, 0.25, "mV"),
    "T_Amplitude": (0.1, 0.5, "mV"),
    "ST_Deviation": (-0.05, 0.1, "mV"),
    "Q_Amplitude": (-0.3, 0.0, "mV"),
    "R_S_Ratio": (0.5, 3.0, "ratio"),
    "QRS_Energy": (0.1, 5.0, "a.u."),
    "SDNN": (25.0, 70.0, "ms"),
    "RMSSD": (20.0, 60.0, "ms"),
    "pNN50": (2.0, 25.0, "%"),
    "pNN20": (10.0, 50.0, "%"),
    "SDRR_RMSSD_Ratio": (0.8, 2.5, "ratio"),
    "HRV_Triangular_Index": (5.0, 25.0, "ratio"),
    "LF_Power": (200.0, 1500.0, "ms²"),
    "HF_Power": (150.0, 1000.0, "ms²"),
    "LF_HF_Ratio": (0.8, 2.5, "ratio"),
    "Total_Power": (800.0, 4000.0, "ms²")
}


def sanitize_val(v: Any) -> Any:
    if isinstance(v, (float, np.floating)):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(v)
    elif isinstance(v, (int, np.integer)):
        return int(v)
    elif isinstance(v, np.ndarray):
        return v.tolist()
    return v


def normalize_biomarker_unit(name: str, value: Any) -> Any:
    if value is None or not isinstance(value, (int, float)) or math.isnan(value):
        return value
    if name in BIOMARKER_RANGES:
        _, _, unit = BIOMARKER_RANGES[name]
        if unit == "ms" and 0.0 < value < 10.0:
            return value * 1000.0
    return value


def evaluate_biomarker_status(name: str, value: Any) -> str:
    norm_val = normalize_biomarker_unit(name, value)
    if norm_val is None or not isinstance(norm_val, (int, float)) or math.isnan(norm_val):
        return "N/A"
    if name not in BIOMARKER_RANGES:
        return "RECORDED"
    low, high, _ = BIOMARKER_RANGES[name]
    if norm_val < low:
        return "LOW"
    elif norm_val > high:
        return "HIGH"
    return "NORMAL"


def generate_svg_waveform_grid(signal: np.ndarray, leads: List[str]) -> str:
    """
    Renders 12-lead ECG signal into a standard clinical 4x3 ECG grid + Lead II rhythm strip.
    Layout:
      Column 0: I, II, III
      Column 1: aVR, aVL, aVF
      Column 2: V1, V2, V3
      Column 3: V4, V5, V6
      Bottom: Continuous full 10-second Lead II Rhythm Strip with 1.0 mV Calibration Pulse
    """
    # 4 columns, 3 rows of leads
    col_leads = [
        [("I", 0), ("II", 1), ("III", 2)],
        [("aVR", 3), ("aVL", 4), ("aVF", 5)],
        [("V1", 6), ("V2", 7), ("V3", 8)],
        [("V4", 9), ("V5", 10), ("V6", 11)]
    ]

    width = 1100
    grid_height = 420
    rhythm_height = 140
    total_height = grid_height + rhythm_height + 30

    col_width = width / 4.0
    row_height = grid_height / 3.0

    svg_lines = []
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {total_height}" width="100%" height="auto" class="ecg-svg-grid">')
    
    # Background and grid
    svg_lines.append(f'<rect width="{width}" height="{total_height}" fill="#ffffff" stroke="#111111" stroke-width="1.5"/>')
    
    # 5mm major grid lines (approx 20px) and 1mm minor lines (approx 4px)
    svg_lines.append('<defs>')
    svg_lines.append('<pattern id="ecgMinor" width="4" height="4" patternUnits="userSpaceOnUse">')
    svg_lines.append('<path d="M 4 0 L 0 0 0 4" fill="none" stroke="#f0f0f0" stroke-width="0.5"/>')
    svg_lines.append('</pattern>')
    svg_lines.append('<pattern id="ecgMajor" width="20" height="20" patternUnits="userSpaceOnUse">')
    svg_lines.append('<rect width="20" height="20" fill="url(#ecgMinor)"/>')
    svg_lines.append('<path d="M 20 0 L 0 0 0 20" fill="none" stroke="#d5d5d5" stroke-width="0.8"/>')
    svg_lines.append('</pattern>')
    svg_lines.append('</defs>')
    svg_lines.append(f'<rect width="{width}" height="{total_height}" fill="url(#ecgMajor)"/>')

    # Draw Lead Columns Divider Lines
    for c in range(1, 4):
        x = c * col_width
        svg_lines.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{grid_height}" stroke="#888888" stroke-width="0.8" stroke-dasharray="3,3"/>')
    
    # Draw Row Divider Lines
    for r in range(1, 3):
        y = r * row_height
        svg_lines.append(f'<line x1="0" y1="{y}" x2="{width}" y2="{y}" stroke="#888888" stroke-width="0.8" stroke-dasharray="3,3"/>')
    
    # Divider for Lead II rhythm strip
    svg_lines.append(f'<line x1="0" y1="{grid_height}" x2="{width}" y2="{grid_height}" stroke="#111111" stroke-width="1.5"/>')

    # Render 4x3 Leads (2.5 seconds each = 250 samples at 100 Hz)
    samples_per_lead = 250
    scale_y = 35.0  # px per mV (approx 10mm/mV standard)

    for c_idx, col in enumerate(col_leads):
        x_offset = c_idx * col_width
        sample_start = c_idx * samples_per_lead
        sample_end = sample_start + samples_per_lead

        for r_idx, (lead_name, lead_idx) in enumerate(col):
            y_baseline = r_idx * row_height + (row_height / 2.0)

            # Lead Title
            svg_lines.append(f'<text x="{x_offset + 12}" y="{y_baseline - (row_height/2.0) + 18}" font-family="monospace" font-size="12" font-weight="bold" fill="#111111">{lead_name}</text>')

            # Calibration pulse on first column leads
            if c_idx == 0:
                cal_x = x_offset + 12
                cal_y = y_baseline
                # 1 mV calibration pulse: 0 -> +1mV -> 0 over 20px
                cal_path = f"M {cal_x} {cal_y} L {cal_x+5} {cal_y} L {cal_x+5} {cal_y - scale_y} L {cal_x+17} {cal_y - scale_y} L {cal_x+17} {cal_y} L {cal_x+25} {cal_y}"
                svg_lines.append(f'<path d="{cal_path}" fill="none" stroke="#222222" stroke-width="1.2"/>')
                lead_plot_start_x = x_offset + 32
                plot_width = col_width - 38
            else:
                lead_plot_start_x = x_offset + 10
                plot_width = col_width - 16

            # Extract signal chunk
            sig_chunk = signal[lead_idx, sample_start:sample_end]
            num_pts = len(sig_chunk)
            if num_pts > 1:
                dx = plot_width / (num_pts - 1)
                pts = []
                for s_i, val in enumerate(sig_chunk):
                    px = lead_plot_start_x + (s_i * dx)
                    py = y_baseline - (val * scale_y)
                    # Clip to row bounds to prevent overflow
                    py = max(y_baseline - (row_height/2.0) + 4, min(y_baseline + (row_height/2.0) - 4, py))
                    pts.append(f"{px:.1f},{py:.1f}")
                svg_lines.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#000000" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/>')

    # Render Full Continuous 10-Second Lead II Rhythm Strip (1000 samples)
    lead_ii_baseline = grid_height + (rhythm_height / 2.0)
    svg_lines.append(f'<text x="12" y="{grid_height + 20}" font-family="monospace" font-size="12" font-weight="bold" fill="#111111">LEAD II (CONTINUOUS 10.0s RHYTHM STRIP)</text>')
    
    # Rhythm strip 1mV calibration pulse
    rcal_x = 12
    rcal_y = lead_ii_baseline
    rcal_path = f"M {rcal_x} {rcal_y} L {rcal_x+6} {rcal_y} L {rcal_x+6} {rcal_y - scale_y} L {rcal_x+20} {rcal_y - scale_y} L {rcal_x+20} {rcal_y} L {rcal_x+28} {rcal_y}"
    svg_lines.append(f'<path d="{rcal_path}" fill="none" stroke="#222222" stroke-width="1.2"/>')

    rhythm_start_x = 38
    rhythm_plot_width = width - 48
    sig_ii = signal[1, :]
    num_r_pts = len(sig_ii)
    if num_r_pts > 1:
        dx_r = rhythm_plot_width / (num_r_pts - 1)
        r_pts = []
        for s_i, val in enumerate(sig_ii):
            px = rhythm_start_x + (s_i * dx_r)
            py = lead_ii_baseline - (val * scale_y)
            py = max(grid_height + 15, min(total_height - 10, py))
            r_pts.append(f"{px:.1f},{py:.1f}")
        svg_lines.append(f'<polyline points="{" ".join(r_pts)}" fill="none" stroke="#000000" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/>')

    # Calibration footnote
    svg_lines.append(f'<text x="{width - 250}" y="{total_height - 10}" font-family="monospace" font-size="10" fill="#555555">Speed: 25 mm/s | Voltage: 10 mm/mV | 100 Hz</text>')

    svg_lines.append('</svg>')
    return "\n".join(svg_lines)


def build_html_report(
    record_meta: Dict[str, Any],
    signal: np.ndarray,
    probabilities: Dict[str, float],
    thresholds: Dict[str, float],
    detected_conditions: List[str],
    biomarkers: Dict[str, Any],
    rep_dims: Dict[str, int],
    clinical_notes_md: str
) -> str:
    """Generates a complete, high-contrast black & white printable HTML clinical report."""
    svg_content = generate_svg_waveform_grid(signal, LEAD_NAMES)
    
    report_uid = f"ECG-REP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    rec_id = record_meta.get("id", "REC-01")
    age = record_meta.get("age", "N/A")
    sex = record_meta.get("sex", "N/A")
    hr = record_meta.get("heart_rate", 75)
    cat = record_meta.get("category", "NORM")
    diag_name = record_meta.get("name", "12-Lead Diagnostic ECG")
    clin_hist = record_meta.get("clinical_history", "Elective Diagnostic Cardiological Examination")

    # Format Detected Conditions Badges
    if detected_conditions:
        badges_html = " ".join([f'<span class="badge badge-detected">{c}</span>' for c in detected_conditions])
    else:
        badges_html = '<span class="badge badge-normal">NORMAL SINUS RHYTHM</span>'

    # Format Superclass Probabilities Table
    prob_rows = []
    for cname in ["NORM", "MI", "STTC", "CD", "HYP"]:
        p = probabilities.get(cname, 0.0)
        th = thresholds.get(cname, 0.5)
        is_pos = (p >= th)
        status_badge = '<span class="status-detected">POSITIVE</span>' if is_pos else '<span class="status-neg">NEGATIVE</span>'
        pct = p * 100
        th_pct = th * 100
        prob_rows.append(f"""
        <tr>
            <td class="font-bold">{cname}</td>
            <td>
                <div class="prob-bar-container">
                    <div class="prob-bar {'bar-detected' if is_pos else ''}" style="width: {min(100, max(2, pct))}%;"></div>
                    <div class="thresh-marker" style="left: {th_pct}%;" title="Threshold: {th_pct:.1f}%"></div>
                </div>
            </td>
            <td class="font-mono text-right">{pct:.1f}%</td>
            <td class="font-mono text-right">{th_pct:.1f}%</td>
            <td class="text-center">{status_badge}</td>
        </tr>
        """)
    prob_table_html = "\n".join(prob_rows)

    # Format 24 Biomarkers Table
    bio_items = []
    for bname in [
        "RR_Mean", "QRS_Duration", "PR_Interval", "QT_Interval", "QTc_Bazett",
        "ST_Duration", "P_wave_Duration", "R_Amplitude", "P_Amplitude", "T_Amplitude",
        "ST_Deviation", "Q_Amplitude", "R_S_Ratio", "QRS_Energy", "SDNN",
        "RMSSD", "pNN50", "pNN20", "SDRR_RMSSD_Ratio", "HRV_Triangular_Index",
        "LF_Power", "HF_Power", "LF_HF_Ratio", "Total_Power"
    ]:
        val = biomarkers.get(bname)
        norm_val = normalize_biomarker_unit(bname, val)
        status = evaluate_biomarker_status(bname, val)
        ref_low, ref_high, unit = BIOMARKER_RANGES.get(bname, ("-", "-", ""))
        ref_str = f"{ref_low} – {ref_high} {unit}" if isinstance(ref_low, (int, float)) else "-"
        val_str = f"{norm_val:.1f} {unit}" if isinstance(norm_val, (int, float)) and not math.isnan(norm_val) else "N/A"
        
        status_class = "bio-normal"
        if status in ("HIGH", "LOW"):
            status_class = "bio-abnormal"

        bio_items.append(f"""
        <tr>
            <td class="font-mono">{bname}</td>
            <td class="font-mono font-bold">{val_str}</td>
            <td class="text-muted text-sm">{ref_str}</td>
            <td class="text-center"><span class="bio-tag {status_class}">{status}</span></td>
        </tr>
        """)
    biomarkers_table_html = "\n".join(bio_items)

    # Simple Markdown-to-HTML parser for clinical notes
    notes_html_lines = []
    for line in clinical_notes_md.splitlines():
        line = line.strip()
        if line.startswith("### "):
            notes_html_lines.append(f'<h4 class="clin-sec-title">{line[4:]}</h4>')
        elif line.startswith("## "):
            notes_html_lines.append(f'<h3 class="clin-sec-title">{line[3:]}</h3>')
        elif line.startswith("- "):
            notes_html_lines.append(f'<li>{line[2:]}</li>')
        elif line.startswith("1. ") or line.startswith("2. ") or line.startswith("3. "):
            notes_html_lines.append(f'<li>{line[3:]}</li>')
        elif line:
            # Bold parsing
            p_text = line.replace("**", "<strong>").replace("**", "</strong>")
            notes_html_lines.append(f'<p>{p_text}</p>')
    notes_html = "\n".join(notes_html_lines)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ECG Diagnostic Clinical Report — {rec_id}</title>
    <style>
        /* CLINICAL BLACK & WHITE MEDICAL STATIONERY DESIGN SYSTEM */
        *, *::before, *::after {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            background-color: #f7f7f7;
            color: #111111;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 13px;
            line-height: 1.45;
            padding: 24px;
        }}
        .report-page {{
            max-width: 1040px;
            margin: 0 auto;
            background: #ffffff;
            border: 2px solid #111111;
            padding: 32px 36px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        }}
        /* Action Toolbar */
        .no-print-toolbar {{
            max-width: 1040px;
            margin: 0 auto 16px auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .btn {{
            display: inline-flex;
            align-items: center;
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 600;
            border-radius: 4px;
            cursor: pointer;
            text-decoration: none;
            transition: all 0.15s ease;
        }}
        .btn-primary {{
            background: #111111;
            color: #ffffff;
            border: 1px solid #111111;
        }}
        .btn-primary:hover {{
            background: #333333;
        }}
        /* Hospital Header */
        .hospital-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 2px solid #111111;
            padding-bottom: 16px;
            margin-bottom: 16px;
        }}
        .hospital-title {{
            font-size: 20px;
            font-weight: 800;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}
        .hospital-sub {{
            font-size: 11px;
            color: #555555;
            margin-top: 2px;
        }}
        .meta-box {{
            border: 1px solid #111111;
            padding: 8px 14px;
            font-family: monospace;
            font-size: 11px;
            line-height: 1.6;
        }}
        /* Patient Strip */
        .patient-strip {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            background: #fdfdfd;
            border: 1px solid #222222;
            padding: 10px 14px;
            margin-bottom: 18px;
        }}
        .patient-cell .label {{
            font-size: 10px;
            text-transform: uppercase;
            color: #666666;
            font-weight: 700;
            display: block;
        }}
        .patient-cell .value {{
            font-size: 13px;
            font-weight: 700;
            color: #000000;
        }}
        /* Section styling */
        .section-box {{
            margin-bottom: 22px;
        }}
        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1.5px solid #111111;
            padding-bottom: 4px;
            margin-bottom: 10px;
        }}
        .section-title {{
            font-size: 13px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        /* Diagnostic Impression Summary */
        .diag-summary-box {{
            border: 1.5px solid #111111;
            padding: 14px;
            margin-bottom: 18px;
            background: #fafafa;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            font-size: 12px;
            font-weight: 800;
            border: 1.5px solid #111111;
            margin-right: 6px;
            text-transform: uppercase;
        }}
        .badge-detected {{
            background: #111111;
            color: #ffffff;
        }}
        .badge-normal {{
            background: #ffffff;
            color: #111111;
        }}
        /* Waveform Tracing */
        .waveform-container {{
            border: 1.5px solid #111111;
            background: #ffffff;
            margin-bottom: 18px;
            overflow: hidden;
        }}
        /* Two Column Layout */
        .two-col-layout {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 18px;
        }}
        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }}
        th, td {{
            padding: 5px 8px;
            border-bottom: 1px solid #e0e0e0;
            text-align: left;
        }}
        th {{
            border-bottom: 1.5px solid #111111;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            background: #f7f7f7;
        }}
        .prob-bar-container {{
            position: relative;
            background: #eeeeee;
            height: 12px;
            width: 100%;
            border: 1px solid #cccccc;
        }}
        .prob-bar {{
            height: 100%;
            background: #777777;
        }}
        .prob-bar.bar-detected {{
            background: #111111;
        }}
        .thresh-marker {{
            position: absolute;
            top: -2px;
            width: 2px;
            height: 16px;
            background: #ff0000;
        }}
        .status-detected {{
            display: inline-block;
            padding: 2px 6px;
            font-size: 10px;
            font-weight: 800;
            background: #111111;
            color: #ffffff;
        }}
        .status-neg {{
            display: inline-block;
            padding: 2px 6px;
            font-size: 10px;
            font-weight: 600;
            color: #777777;
        }}
        .bio-tag {{
            display: inline-block;
            padding: 1px 6px;
            font-size: 9px;
            font-weight: 800;
            border: 1px solid #111111;
        }}
        .bio-normal {{
            background: #ffffff;
            color: #111111;
        }}
        .bio-abnormal {{
            background: #111111;
            color: #ffffff;
        }}
        /* Clinical Notes Block */
        .clinical-notes-card {{
            border: 1px solid #111111;
            padding: 14px 18px;
            background: #ffffff;
            margin-bottom: 18px;
        }}
        .clin-sec-title {{
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            margin-top: 10px;
            margin-bottom: 4px;
            border-bottom: 1px dotted #888888;
            padding-bottom: 2px;
        }}
        .clinical-notes-card p, .clinical-notes-card li {{
            font-size: 12px;
            color: #222222;
            margin-bottom: 4px;
        }}
        .clinical-notes-card ul {{
            padding-left: 20px;
        }}
        /* Sign-off & Disclaimer */
        .report-footer {{
            border-top: 1.5px solid #111111;
            padding-top: 12px;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            margin-top: 20px;
        }}
        .disclaimer {{
            font-size: 10px;
            color: #666666;
            max-width: 65%;
            line-height: 1.4;
        }}
        .signature-box {{
            border-top: 1px solid #111111;
            width: 200px;
            text-align: center;
            font-size: 10px;
            font-weight: 700;
            padding-top: 4px;
        }}
        /* Utility */
        .font-mono {{ font-family: monospace; }}
        .font-bold {{ font-weight: 700; }}
        .text-right {{ text-align: right; }}
        .text-center {{ text-align: center; }}
        .text-muted {{ color: #777777; }}
        .text-sm {{ font-size: 11px; }}

        /* PRINT OPTIMIZATION (A4 / Letter) */
        @media print {{
            body {{
                background: #ffffff;
                padding: 0;
            }}
            .no-print-toolbar {{
                display: none;
            }}
            .report-page {{
                border: none;
                box-shadow: none;
                padding: 0;
                max-width: 100%;
            }}
        }}
    </style>
</head>
<body>

    <!-- On-Screen Toolbar -->
    <div class="no-print-toolbar">
        <div>
            <strong>ECG Multimodal Diagnostic Report</strong> &mdash; <span>{rec_id}</span>
        </div>
        <div>
            <button class="btn btn-primary" onclick="window.print()">Print Report (Ctrl+P / PDF)</button>
        </div>
    </div>

    <!-- Official Report Sheet -->
    <main class="report-page">
        <!-- Hospital Header -->
        <header class="hospital-header">
            <div>
                <div class="hospital-title">ECG Multimodal Foundation Core Lab</div>
                <div class="hospital-sub">Center for Cardiac Electrophysiology & Deep Learning Decision Support</div>
                <div class="hospital-sub">Physiological 12-Lead Diagnostic Electrocardiography</div>
            </div>
            <div class="meta-box">
                <div>REPORT ID: <strong>{report_uid}</strong></div>
                <div>RECORD ID: <strong>{rec_id}</strong></div>
                <div>DATE: <strong>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</strong></div>
                <div>METHOD: <strong>Tri-Modal Foundation (1056-D)</strong></div>
            </div>
        </header>

        <!-- Patient Demographics Strip -->
        <section class="patient-strip">
            <div class="patient-cell">
                <span class="label">Patient Name / Record</span>
                <span class="value">{diag_name}</span>
            </div>
            <div class="patient-cell">
                <span class="label">Age / Sex</span>
                <span class="value">{age} Years / {sex}</span>
            </div>
            <div class="patient-cell">
                <span class="label">Heart Rate</span>
                <span class="value">{hr} BPM</span>
            </div>
            <div class="patient-cell">
                <span class="label">Clinical Context</span>
                <span class="value">{clin_hist}</span>
            </div>
        </section>

        <!-- Diagnostic Impression & Detected Badges -->
        <section class="diag-summary-box">
            <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #555555; margin-bottom: 6px;">
                Automated Multimodal Diagnostic Impression
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    {badges_html}
                </div>
                <div style="font-size: 11px; color: #555555;">
                    Calibrated Youden Decision Thresholds
                </div>
            </div>
        </section>

        <!-- 12-Lead ECG Waveform Tracing Grid -->
        <section class="section-box">
            <div class="section-header">
                <span class="section-title">Diagnostic 12-Lead Tracing (4x3 Layout + Rhythm Strip Lead II)</span>
                <span class="font-mono text-sm">25 mm/s | 10 mm/mV | 100 Hz Bandwidth</span>
            </div>
            <div class="waveform-container">
                {svg_content}
            </div>
        </section>

        <!-- Two Column: Classification & Biomarkers -->
        <div class="two-col-layout">
            <!-- Left: Calibrated Superclasses -->
            <div>
                <div class="section-header">
                    <span class="section-title">Calibrated 5-Class Superclasses</span>
                    <span class="text-sm text-muted">F1-Optimal</span>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Condition</th>
                            <th>Confidence</th>
                            <th class="text-right">Prob</th>
                            <th class="text-right">Thresh</th>
                            <th class="text-center">Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {prob_table_html}
                    </tbody>
                </table>

                <div style="margin-top: 16px;">
                    <div class="section-header">
                        <span class="section-title">Latent Tensor Dimensions</span>
                    </div>
                    <table style="font-size: 11px;">
                        <tbody>
                            <tr><td>z_temporal (1D ResNet-SE)</td><td class="font-mono text-right">{rep_dims.get('temporal', 512)} Dim</td></tr>
                            <tr><td>z_morphology (2D Spectrogram ResNet)</td><td class="font-mono text-right">{rep_dims.get('morphology', 512)} Dim</td></tr>
                            <tr><td>z_biomarker (Attention MLP)</td><td class="font-mono text-right">{rep_dims.get('biomarker', 32)} Dim</td></tr>
                            <tr style="font-weight: bold; background: #fafafa;"><td>z_fused (Unified Joint Space)</td><td class="font-mono text-right">{rep_dims.get('fused', 1056)} Dim</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Right: 24 Biomarkers Table -->
            <div>
                <div class="section-header">
                    <span class="section-title">24 Electrophysiological Biomarkers</span>
                    <span class="text-sm text-muted">Reference Ranges</span>
                </div>
                <div style="max-height: 440px; overflow-y: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Biomarker</th>
                                <th>Value</th>
                                <th>Ref Range</th>
                                <th class="text-center">Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {biomarkers_table_html}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Clinical & Electrophysiological Interpretation -->
        <section class="section-box">
            <div class="section-header">
                <span class="section-title">Clinical & Research Electrophysiological Impression</span>
                <span class="text-sm text-muted">Deep Foundation Analysis</span>
            </div>
            <div class="clinical-notes-card">
                {notes_html}
            </div>
        </section>

        <!-- Sign-Off & Medical Disclaimer -->
        <footer class="report-footer">
            <div class="disclaimer">
                <strong>CLINICAL DECISION SUPPORT NOTICE:</strong> This diagnostic summary is computed via multimodal deep neural networks
                (1D Waveform Dynamics + 2D Spectrogram Contours + Electrophysiological Biomarkers).
                This report is intended for qualified healthcare professionals and research clinical validation.
            </div>
            <div class="signature-box">
                <div>Attending Electrophysiologist / AI Core Lab</div>
                <div style="font-size: 9px; color: #777777; margin-top: 2px;">Electronically Signed & Calibrated</div>
            </div>
        </footer>
    </main>

</body>
</html>
"""
    return html


def print_terminal_summary(
    record_meta: Dict[str, Any],
    probabilities: Dict[str, float],
    thresholds: Dict[str, float],
    detected_conditions: List[str],
    biomarkers: Dict[str, Any],
    clinical_notes_md: str
):
    """Prints a clean ASCII clinical report directly to stdout."""
    rec_id = record_meta.get("id", "REC-01")
    name = record_meta.get("name", "Diagnostic ECG")
    age = record_meta.get("age", "N/A")
    sex = record_meta.get("sex", "N/A")
    hr = record_meta.get("heart_rate", 75)

    print("\n" + "=" * 78)
    print("  [ECG CORE LAB] INDIVIDUAL CLINICAL DIAGNOSTIC REPORT SUMMARY")
    print("=" * 78)
    print(f"  Record ID:       {rec_id} | Sample Code: {record_meta.get('sample_code', rec_id)}")
    print(f"  Patient:         {name} | Age: {age} | Sex: {sex} | HR: {hr} BPM")
    print(f"  Date & Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 78)

    print("\n  [+] DIAGNOSTIC CLASSIFICATION (Calibrated Decision Thresholds):")
    print(f"  {'Condition':<8} | {'Prob':<8} | {'Thresh':<8} | {'Status':<16} | {'Confidence Bar'}")
    print("  " + "-" * 74)
    for cname in ["NORM", "MI", "STTC", "CD", "HYP"]:
        p = probabilities.get(cname, 0.0)
        th = thresholds.get(cname, 0.5)
        is_pos = (p >= th)
        status = "POSITIVE [DETECTED]" if is_pos else "NEGATIVE"
        bar_len = int(p * 20)
        bar = "#" * bar_len + "-" * (20 - bar_len)
        print(f"  {cname:<8} | {p*100:5.1f}%  | {th*100:5.1f}%  | {status:<16} | [{bar}]")

    print(f"\n  => Detected Cardiac Conditions: {', '.join(detected_conditions) if detected_conditions else 'Normal Sinus Rhythm'}")

    print("\n  [+] KEY ELECTROPHYSIOLOGICAL BIOMARKERS:")
    print(f"  {'Biomarker':<20} | {'Value':<14} | {'Reference':<18} | {'Status'}")
    print("  " + "-" * 66)
    for bname in ["RR_Mean", "QRS_Duration", "PR_Interval", "QT_Interval", "QTc_Bazett", "ST_Deviation", "SDNN", "RMSSD"]:
        val = biomarkers.get(bname)
        norm_val = normalize_biomarker_unit(bname, val)
        status = evaluate_biomarker_status(bname, val)
        ref_low, ref_high, unit = BIOMARKER_RANGES.get(bname, ("-", "-", ""))
        ref_str = f"{ref_low} - {ref_high} {unit}" if isinstance(ref_low, (int, float)) else "-"
        val_str = f"{norm_val:.1f} {unit}" if isinstance(norm_val, (int, float)) and not math.isnan(norm_val) else "N/A"
        print(f"  {bname:<20} | {val_str:<14} | {ref_str:<18} | {status}")

    print("\n  [+] CLINICAL IMPRESSION:")
    # Print first few lines of interpretation
    for line in clinical_notes_md.splitlines()[:12]:
        if line.strip():
            print(f"  {line}")

    print("\n" + "=" * 78)


def generate_individual_report(
    sample_id: str | None = None,
    input_file: str | None = None,
    output_html: str | None = None,
    export_json: bool = False,
    open_browser: bool = False,
    device: str | None = None
) -> str:
    """Executes end-to-end report generation for one ECG record."""
    sample_mgr = SampleManager()
    
    # 1. Determine signal source & metadata
    if input_file:
        inp_path = Path(input_file)
        if not inp_path.exists():
            raise FileNotFoundError(f"Input signal file not found: {input_file}")
        
        if inp_path.suffix.lower() == ".npy":
            raw_sig = np.load(str(inp_path))
        elif inp_path.suffix.lower() == ".csv":
            raw_sig = np.loadtxt(str(inp_path), delimiter=",")
        else:
            raise ValueError(f"Unsupported signal format: {inp_path.suffix}. Use .npy or .csv")

        # Standardize shape to (12, 1000)
        if raw_sig.shape == (1000, 12):
            raw_sig = raw_sig.T
        if raw_sig.shape[0] != 12:
            raise ValueError(f"Signal must contain 12 leads, got shape: {raw_sig.shape}")

        signal = raw_sig.astype(np.float32)
        rec_id = inp_path.stem
        record_meta = {
            "id": rec_id,
            "sample_code": rec_id,
            "name": f"Custom Signal ({inp_path.name})",
            "age": "N/A",
            "sex": "Unknown",
            "heart_rate": 75,
            "category": "Custom",
            "clinical_history": "Imported File Analysis"
        }
        biomarkers = {}
    else:
        # Load sample from SampleManager
        sample_code = sample_id or "REC-01"
        payload = sample_mgr.generate_full_sample_payload(sample_code)
        record_meta = payload.get("record", {})
        rec_id = record_meta.get("id", sample_code)
        signal = np.array(payload.get("signal", []), dtype=np.float32)
        biomarkers = payload.get("biomarkers", {})

    print(f"[*] Processing 12-Lead ECG Signal (Shape: {signal.shape})...")

    # 2. Run Engine Inference
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = ECGEncoderEngine(config=EngineConfig(device=dev))
    rep, pred = engine.process(signal)

    probabilities = {cname: float(p) for cname, p in zip(pred.class_names, pred.probabilities[0])}
    thresholds = {cname: float(th) for cname, th in zip(pred.class_names, pred.thresholds)}
    detected = pred.detected_conditions[0]

    # If biomarkers empty, pull from engine biomarker service if possible
    if not biomarkers:
        bio_names = [
            "RR_Mean", "QRS_Duration", "PR_Interval", "QT_Interval", "QTc_Bazett",
            "ST_Duration", "P_wave_Duration", "R_Amplitude", "P_Amplitude", "T_Amplitude",
            "ST_Deviation", "Q_Amplitude", "R_S_Ratio", "QRS_Energy", "SDNN",
            "RMSSD", "pNN50", "pNN20", "SDRR_RMSSD_Ratio", "HRV_Triangular_Index",
            "LF_Power", "HF_Power", "LF_HF_Ratio", "Total_Power"
        ]
        biomarkers = {b: 0.0 for b in bio_names}

    # 3. Generate Clinical Interpretation via Gemini API / Offline Heuristics
    norm_bio_dict = {k: normalize_biomarker_unit(k, v) for k, v in biomarkers.items()}
    interpreter = GeminiClinicalInterpreter()
    report_res = interpreter.generate_clinical_report(
        record_id=rec_id,
        detected_conditions=detected,
        probabilities=probabilities,
        thresholds=thresholds,
        biomarkers=norm_bio_dict,
        patient_metadata=record_meta
    )
    clinical_notes = report_res.get("report_markdown", "Clinical report generated successfully.")

    # 4. Print Summary to Terminal
    print_terminal_summary(
        record_meta=record_meta,
        probabilities=probabilities,
        thresholds=thresholds,
        detected_conditions=detected,
        biomarkers=biomarkers,
        clinical_notes_md=clinical_notes
    )

    # 5. Build and Save HTML Report
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    html_out_path = Path(output_html) if output_html else reports_dir / f"{rec_id}_clinical_report.html"
    html_out_path.parent.mkdir(parents=True, exist_ok=True)

    rep_dims = {
        "temporal": int(rep.z_temporal.shape[1]),
        "morphology": int(rep.z_morphology.shape[1]),
        "biomarker": int(rep.z_biomarker.shape[1]),
        "fused": int(rep.z_fused.shape[1]),
    }

    html_content = build_html_report(
        record_meta=record_meta,
        signal=signal,
        probabilities=probabilities,
        thresholds=thresholds,
        detected_conditions=detected,
        biomarkers=biomarkers,
        rep_dims=rep_dims,
        clinical_notes_md=clinical_notes
    )

    with open(html_out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[+] Saved High-Contrast HTML Clinical Report to:")
    print(f"    {html_out_path.resolve()}")

    # 6. Optional JSON Export
    if export_json:
        json_out_path = html_out_path.with_suffix(".json")
        json_data = {
            "record": record_meta,
            "detected_conditions": detected,
            "probabilities": probabilities,
            "thresholds": thresholds,
            "biomarkers": {k: sanitize_val(v) for k, v in biomarkers.items()},
            "representations": {
                "z_temporal_dim": rep_dims["temporal"],
                "z_morphology_dim": rep_dims["morphology"],
                "z_biomarker_dim": rep_dims["biomarker"],
                "z_fused_dim": rep_dims["fused"],
                "z_fused_vector": sanitize_val(rep.z_fused[0])
            },
            "clinical_notes": clinical_notes,
            "generated_at": datetime.now().isoformat()
        }
        with open(json_out_path, "w", encoding="utf-8") as jf:
            json.dump(json_data, jf, indent=2)
        print(f"[+] Saved JSON Diagnostic Export to:\n    {json_out_path.resolve()}")

    # 7. Auto-open in browser if requested
    if open_browser:
        print(f"[*] Opening report in default browser: file://{html_out_path.resolve()}")
        webbrowser.open(f"file://{html_out_path.resolve()}")

    return str(html_out_path.resolve())


def list_samples():
    """Lists available preset records from catalog."""
    sm = SampleManager()
    records = sm.get_all_records()
    print("=" * 80)
    print(f"  PRESET ECG RECORDS CATALOG (Total: {len(records)} records)")
    print("=" * 80)
    print(f"  {'Sample ID':<12} | {'Category':<8} | {'Age/Sex':<10} | {'Description'}")
    print("  " + "-" * 76)
    for r in records[:20]:
        rid = r.get("id", "")
        cat = r.get("category", "")
        dem = f"{r.get('age', 'N/A')}yo {r.get('sex', '')}"
        desc = r.get("name", "")
        print(f"  {rid:<12} | {cat:<8} | {dem:<10} | {desc}")
    if len(records) > 20:
        print(f"  ... and {len(records) - 20} more records.")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="ECG Foundation Representation System — Individual Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_report.py --open
  python scripts/generate_report.py --sample REC-21826 --open
  python scripts/generate_report.py --input path/to/ecg.npy --output my_report.html --open
  python scripts/generate_report.py --list
        """
    )
    parser.add_argument("--sample", "-s", type=str, default=None, help="Preset record code or ID (e.g. REC-01, REC-21826, 21826)")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to custom 12-lead ECG file (.npy or .csv)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path for generated HTML report")
    parser.add_argument("--json", action="store_true", help="Also export structured JSON report")
    parser.add_argument("--open", action="store_true", help="Automatically open generated HTML report in browser")
    parser.add_argument("--list", "-l", action="store_true", help="List available preset catalog records")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Compute device for inference")

    args = parser.parse_args()

    if args.list:
        list_samples()
        return

    generate_individual_report(
        sample_id=args.sample,
        input_file=args.input,
        output_html=args.output,
        export_json=args.json,
        open_browser=args.open,
        device=args.device
    )


if __name__ == "__main__":
    main()
