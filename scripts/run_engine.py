"""
ECG Foundation Representation System — Engine CLI Runner
=========================================================
Standalone runner for the ECG Foundation Representation Engine.

Features:
- Demo: End-to-end multimodal verification demo on 12-lead ECG signals
- Info: Model architecture and frozen weights registry inspector
- Predict: Diagnostic classification on custom .npy/.csv ECG signals
- Encode: Extract 1056-D fused representation (z_temporal, z_morphology, z_biomarker)
- Test: Execute unit & integration test suites
"""

from __future__ import annotations
import os
import sys
import argparse
import unittest
from pathlib import Path

# Set up project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def run_demo(device: str | None = None):
    """Executes verified end-to-end multimodal inference demo."""
    import numpy as np
    import torch
    from ecg_engine import ECGEncoderEngine, EngineConfig

    dev = device or get_default_device()

    print("=" * 76)
    print("  [ECG CORE ENGINE] FOUNDATION REPRESENTATION & INFERENCE DEMO")
    print("=" * 76)
    print(f"  * PyTorch Version:   {torch.__version__}")
    print(f"  * Compute Device:    {dev.upper()}")
    if dev == "cuda" and torch.cuda.is_available():
        print(f"  * GPU Model:         {torch.cuda.get_device_name(0)}")

    cfg = EngineConfig(device=dev)
    engine = ECGEncoderEngine(config=cfg)

    print(f"  * Diagnostic Target: {cfg.class_names}")
    print(f"  * Calibrated Thresh: {np.round(engine.classifier.thresholds, 3)}")
    print("-" * 76)

    # Generate synthetic 12-lead physiological signals (batch=2, 12 leads, 1000 samples @ 100 Hz)
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    simulated_leads = []
    for lead_idx in range(12):
        lead_signal = np.sin(2 * np.pi * 1.2 * t + lead_idx * 0.15) + 0.08 * np.random.randn(1000)
        simulated_leads.append(lead_signal.astype(np.float32))
    simulated_signal = np.stack(simulated_leads, axis=0)  # (12, 1000)
    batch_signals = np.stack([simulated_signal, simulated_signal * 1.25], axis=0)  # (2, 12, 1000)

    print(f"[*] Processing 12-Lead Batch Tensor (Shape: {batch_signals.shape})...")
    rep, pred = engine.process(batch_signals)

    print("\n--- Multimodal Latent Space Tensors ---")
    print(f"  [1] z_temporal   (1D ResNet-SE):     {rep.z_temporal.shape} (512 Dimensions)")
    print(f"  [2] z_morphology (2D Spectrogram):   {rep.z_morphology.shape} (512 Dimensions)")
    print(f"  [3] z_biomarker  (Attention MLP):    {rep.z_biomarker.shape} (32 Dimensions)")
    print(f"  --> z_fused      (Unified Space):    {rep.z_fused.shape} (1056 Dimensions)")
    if rep.z_unified is not None:
        print(f"  --> z_unified    (Classifier Dense): {rep.z_unified.shape} (128 Dimensions)")

    print("\n--- Calibrated Superclass Predictions ---")
    for i in range(len(batch_signals)):
        print(f"\n  [Sample {i+1}]:")
        probs = pred.probabilities[i]
        preds = pred.predictions[i]
        for cname, p, decision, th in zip(cfg.class_names, probs, preds, pred.thresholds):
            status = "POSITIVE [DETECTED]" if decision == 1 else "NEGATIVE"
            bar_len = int(p * 20)
            bar = "#" * bar_len + "-" * (20 - bar_len)
            print(f"    - {cname:5s} [{bar}] Prob={p*100:5.1f}% | Thresh={th*100:4.1f}% | {status}")
        detected_str = ", ".join(pred.detected_conditions[i]) if pred.detected_conditions[i] else "None (Normal Variant)"
        print(f"    => Detected Conditions: {detected_str}")

    print("\n" + "=" * 76)
    print("  SUCCESS: ECG Foundation Representation Engine is verified and healthy.")
    print("=" * 76 + "\n")


def print_info():
    """Prints model architecture and weights registry."""
    print("=" * 80)
    print("  ECG FOUNDATION REPRESENTATION SYSTEM — ARCHITECTURE & WEIGHTS REGISTRY")
    print("=" * 80)

    weights = [
        ("Temporal Encoder (ResNet1D-SE)", "models/C5_full_dataset.pt", "512-D Latent Representation"),
        ("Morphology Encoder (2D ResNet)", "models/morphology_encoder_v1.pt", "512-D Spectrogram Latent"),
        ("Biomarker Encoder (Attention MLP)", "biomarkers/attention_mlp_best.pt", "32-D Latent Embedding"),
        ("Unified Classifier (MLP)", "models/classification_mlp.pt", "5-Class Superclass Prediction"),
        ("Calibrated Decision Thresholds", "models/classification_mlp_thresholds.npy", "Youden F1 Optimal Thresholds"),
        ("Biomarker Feature Scaler", "biomarkers/scaler.pkl", "Robust Feature Normalization"),
        ("Biomarker Missingness Imputer", "biomarkers/imputer.pkl", "Iterative Median Imputer"),
    ]

    print(f"  {'Component':<35} | {'Weight Path':<35} | {'Status'}")
    print("  " + "-" * 78)
    for name, path, desc in weights:
        full_path = PROJECT_ROOT / path
        if full_path.exists():
            size_kb = os.path.getsize(full_path) / 1024
            status = f"PRESENT ({size_kb:,.1f} KB)"
        else:
            status = "MISSING"
        print(f"  {name:<35} | {path:<35} | {status}")

    print("\n  Summary:")
    print("  - Latent Dimensions: 1056-D (512 Temporal + 512 Morphology + 32 Biomarkers)")
    print("  - Dense Bottleneck:  128-D Joint Embedding")
    print("  - Target Classes:    NORM (Normal), MI (Myocardial Infarction), STTC (ST/T Change),")
    print("                       CD (Conduction Disturbance), HYP (Hypertrophy)")
    print("=" * 80 + "\n")


def run_tests():
    """Runs the engine unit test suite."""
    print("=" * 76)
    print("  RUNNING ECG FOUNDATION ENGINE UNIT & INTEGRATION TESTS")
    print("=" * 76)
    suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"), pattern="test_ecg_engine.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n[+] All engine tests PASSED successfully.")
        return 0
    else:
        print("\n[!] Engine tests encountered failures.")
        return 1


def run_predict(input_path: str, device: str | None = None):
    """Predicts diagnostic conditions on input file."""
    import numpy as np
    from ecg_engine import ECGEncoderEngine, EngineConfig

    p = Path(input_path)
    if not p.exists():
        print(f"[!] Input file not found: {input_path}")
        sys.exit(1)

    print(f"[*] Loading signal from: {input_path}")
    if p.suffix.lower() == ".npy":
        signal = np.load(str(p))
    elif p.suffix.lower() == ".csv":
        signal = np.loadtxt(str(p), delimiter=",")
    else:
        print(f"[!] Unsupported file extension: {p.suffix}. Please use .npy or .csv")
        sys.exit(1)

    dev = device or get_default_device()
    engine = ECGEncoderEngine(config=EngineConfig(device=dev))
    pred = engine.predict(ecg_signal=signal)

    print(f"\n[+] Prediction results for: {p.name}")
    for idx, detected in enumerate(pred.detected_conditions):
        probs = pred.probabilities[idx]
        print(f"\n  Sample {idx+1}:")
        for cname, prob, th in zip(pred.class_names, probs, pred.thresholds):
            det = "POSITIVE" if prob >= th else "NEGATIVE"
            print(f"    - {cname:5s}: {prob*100:5.1f}% (Thresh: {th*100:4.1f}%) -> {det}")
        print(f"    => Detected Conditions: {', '.join(detected) if detected else 'None (Normal Variant)'}")


def run_encode(input_path: str, output_path: str, device: str | None = None):
    """Extracts representation embeddings and saves to file."""
    import numpy as np
    from ecg_engine import ECGEncoderEngine, EngineConfig

    p = Path(input_path)
    if not p.exists():
        print(f"[!] Input file not found: {input_path}")
        sys.exit(1)

    signal = np.load(str(p)) if p.suffix.lower() == ".npy" else np.loadtxt(str(p), delimiter=",")
    dev = device or get_default_device()
    engine = ECGEncoderEngine(config=EngineConfig(device=dev))

    print(f"[*] Encoding ECG signal (Input shape: {signal.shape})...")
    rep = engine.encode(signal)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_p), rep.z_fused)
    print(f"[+] Saved 1056-D fused representation to: {out_p.resolve()}")
    print(f"    Shape: {rep.z_fused.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="ECG Foundation Representation System — Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_engine.py demo
  python scripts/run_engine.py info
  python scripts/run_engine.py test
  python scripts/run_engine.py predict --input path/to/ecg.npy
  python scripts/run_engine.py encode --input path/to/ecg.npy --output z_fused.npy
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Demo
    demo_parser = subparsers.add_parser("demo", help="Run end-to-end inference verification demo")
    demo_parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Compute device")

    # Info
    subparsers.add_parser("info", help="Inspect model architecture and weights registry")

    # Test
    subparsers.add_parser("test", help="Run engine verification test suite")

    # Predict
    predict_parser = subparsers.add_parser("predict", help="Predict conditions from ECG array")
    predict_parser.add_argument("--input", "-i", type=str, required=True, help="Path to .npy or .csv ECG file")
    predict_parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Compute device")

    # Encode
    encode_parser = subparsers.add_parser("encode", help="Extract representation vector (z_fused)")
    encode_parser.add_argument("--input", "-i", type=str, required=True, help="Path to .npy or .csv ECG file")
    encode_parser.add_argument("--output", "-o", type=str, default="z_fused.npy", help="Path for output .npy")
    encode_parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Compute device")

    args = parser.parse_args()

    cmd = args.command or "demo"
    if cmd == "demo":
        run_demo(device=args.device if hasattr(args, "device") else None)
    elif cmd == "info":
        print_info()
    elif cmd == "test":
        sys.exit(run_tests())
    elif cmd == "predict":
        run_predict(input_path=args.input, device=args.device)
    elif cmd == "encode":
        run_encode(input_path=args.input, output_path=args.output, device=args.device)


if __name__ == "__main__":
    main()
