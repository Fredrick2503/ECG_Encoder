"""
ECG Foundation Representation System - Main Entry Point & CLI
=============================================================
Unified interface to access the frozen ECG multimodal representation pipeline and engine.

Usage:
    # Run full demonstration & health check
    python main.py demo

    # Print engine architecture and model weights registry
    python main.py info

    # Encode ECG record/signal
    python main.py encode --input path/to/ecg.npy --output path/to/z_fused.npy

    # Predict diagnostic conditions
    python main.py predict --input path/to/ecg.npy
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_demo(device: str = "cpu"):
    """Executes an end-to-end inference verification using real or simulated 12-lead ECG signals."""
    import numpy as np
    import torch
    from ecg_engine import ECGEncoderEngine, EngineConfig
    print("=" * 70)
    print("ECG FOUNDATION REPRESENTATION ENGINE — DEMO & VERIFICATION")
    print("=" * 70)
    
    cfg = EngineConfig(device=device)
    engine = ECGEncoderEngine(config=cfg)
    print(f"[*] Initialized Engine on device: {engine.device}")
    print(f"[*] Diagnostic Classes: {cfg.class_names}")
    print(f"[*] Calibrated Thresholds: {np.round(engine.classifier.thresholds, 3)}")
    
    # Generate test 12-lead signal (batch_size=2, 12 leads, 1000 samples)
    np.random.seed(42)
    # Synthetic clean sinus rhythm proxy with simulated QRS and noise
    t = np.linspace(0, 10, 1000)
    simulated_leads = []
    for lead_idx in range(12):
        lead_signal = np.sin(2 * np.pi * 1.2 * t + lead_idx * 0.1) + 0.1 * np.random.randn(1000)
        simulated_leads.append(lead_signal)
    simulated_signal = np.stack(simulated_leads, axis=0) # (12, 1000)
    
    batch_signals = np.stack([simulated_signal, simulated_signal * 1.2], axis=0) # (2, 12, 1000)
    
    print(f"\n[*] Processing Batch of 12-Lead ECG Signals (Shape: {batch_signals.shape})...")
    rep, pred = engine.process(batch_signals)
    
    print("\n--- Multimodal Representation Output ---")
    print(f"  • Z_temporal   (1D ResNet-SE):    {rep.z_temporal.shape} (Dim: 512)")
    print(f"  • Z_morphology (2D Spectrogram):  {rep.z_morphology.shape} (Dim: 512)")
    print(f"  • Z_biomarker  (Attention MLP):   {rep.z_biomarker.shape} (Dim: 32)")
    print(f"  • Z_fused      (Unified Space):   {rep.z_fused.shape} (Dim: 1056)")
    
    print("\n--- Diagnostic Classification Output ---")
    for i in range(len(batch_signals)):
        print(f"\n[Sample {i+1}]:")
        probs = pred.probabilities[i]
        preds = pred.predictions[i]
        for cname, p, decision, th in zip(cfg.class_names, probs, preds, pred.thresholds):
            status = "POSITIVE [DETECTED]" if decision == 1 else "NEGATIVE"
            print(f"  - {cname:5s}: Prob = {p*100:5.1f}% | Thresh = {th*100:4.1f}% | {status}")
        print(f"  => Detected Conditions: {', '.join(pred.detected_conditions[i])}")
        
    print("\n" + "=" * 70)
    print("SUCCESS: ECG Encoder Engine pipeline is active and verified.")
    print("=" * 70)


def print_info():
    """Prints registry information on frozen weights, models, and project architecture."""
    print("=" * 70)
    print("ECG FOUNDATION REPRESENTATION SYSTEM — ARCHITECTURE & REGISTRY")
    print("=" * 70)
    
    weights = [
        ("Temporal Encoder (ResNet1D-SE)", "models/C5_full_dataset.pt", "512-D Latent Representation"),
        ("Morphology Encoder (2D ResNet)", "models/morphology_encoder_v1.pt", "512-D Spectrogram Latent"),
        ("Biomarker Encoder (Attention MLP)", "biomarkers/attention_mlp_best.pt", "32-D Latent Embedding"),
        ("Unified Classifier (MLP)", "models/classification_mlp.pt", "5-Class Superclass Prediction"),
        ("Calibrated Thresholds", "models/classification_mlp_thresholds.npy", "Optimal F1 Class Thresholds"),
        ("Biomarker Preprocessing Scaler", "biomarkers/scaler.pkl", "Robust Feature Normalization"),
        ("Biomarker Missingness Imputer", "biomarkers/imputer.pkl", "Iterative Median Imputer"),
    ]
    
    print(f"{'Component':<35} | {'Weight / Artifact Path':<35} | {'Status'}")
    print("-" * 85)
    for name, path, desc in weights:
        full_path = PROJECT_ROOT / path
        status = "FOUND" if full_path.exists() else "MISSING"
        size_str = f" ({os.path.getsize(full_path)/1024:.1f} KB)" if full_path.exists() else ""
        print(f"{name:<35} | {path:<35} | {status}{size_str}")
        
    print("\nTotal Multimodal Latent Space: 1056 dimensions (512 Temporal + 512 Morphology + 32 Biomarker)")
    print("Target Diagnostic Classes: NORM (Normal), MI (Myocardial Infarction), STTC (ST/T Change), CD (Conduction Disturbance), HYP (Hypertrophy)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="ECG Foundation Representation System Engine Entrypoint"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Demo command
    subparsers.add_parser("demo", help="Run full pipeline verification demo")
    
    # Info command
    subparsers.add_parser("info", help="Print model registry and architecture info")
    
    # Encode command
    encode_parser = subparsers.add_parser("encode", help="Extract representation vector from ECG file")
    encode_parser.add_argument("--input", "-i", type=str, required=True, help="Path to .npy array or signal file")
    encode_parser.add_argument("--output", "-o", type=str, default="z_fused.npy", help="Output path for z_fused")
    encode_parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict diagnostic conditions from ECG file")
    predict_parser.add_argument("--input", "-i", type=str, required=True, help="Path to .npy array or signal file")
    predict_parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    
    args = parser.parse_args()
    
    if args.command == "demo" or args.command is None:
        import torch
        device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
        run_demo(device=device)
    elif args.command == "info":
        print_info()
    elif args.command == "encode":
        import numpy as np
        from ecg_engine import ECGEncoderEngine, EngineConfig
        engine = ECGEncoderEngine(config=EngineConfig(device=args.device))
        signal = np.load(args.input)
        rep = engine.encode(signal)
        np.save(args.output, rep.z_fused)
        print(f"Saved fused representation to {args.output} (Shape: {rep.z_fused.shape})")
    elif args.command == "predict":
        import numpy as np
        from ecg_engine import ECGEncoderEngine, EngineConfig
        engine = ECGEncoderEngine(config=EngineConfig(device=args.device))
        signal = np.load(args.input)
        pred = engine.predict(ecg_signal=signal)
        print("\nPredictions:")
        for idx, detected in enumerate(pred.detected_conditions):
            print(f"Sample {idx+1}: {', '.join(detected)}")


if __name__ == "__main__":
    main()
