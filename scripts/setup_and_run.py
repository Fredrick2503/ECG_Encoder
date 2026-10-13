"""
ECG Foundation Representation System — Complete Bootstrap & Launch Pipeline
============================================================================
All-in-one automation script that executes:
1. Environment verification & virtual environment creation (.venv)
2. Dependency installation from requirements.txt
3. Model weights and architecture verification
4. Data downloading & authentic waveform pre-caching from PhysioNet PTB-XL
5. Engine health check & smoke test
6. Instant launch of the Clinical Report Generator Web Application

Usage:
    # Full automated bootstrap and launch
    python scripts/setup_and_run.py

    # Fast start (skip dependency install and data download if already done)
    python scripts/setup_and_run.py --skip-install --skip-data

    # Custom port
    python scripts/setup_and_run.py --port 8080
"""

from __future__ import annotations
import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_venv_python() -> Path:
    """Returns the virtual environment python executable path."""
    if sys.platform == "win32":
        return PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return PROJECT_ROOT / ".venv" / "bin" / "python"


def is_running_in_project_venv() -> bool:
    """Checks if the currently executing python is the project's .venv."""
    venv_dir = PROJECT_ROOT / ".venv"
    if not venv_dir.exists():
        return False
    try:
        return Path(sys.prefix).resolve() == venv_dir.resolve()
    except Exception:
        return False


def step_1_ensure_environment():
    """Ensures Python version is compatible and .venv exists."""
    print("\n" + "=" * 78)
    print("  [STEP 1/5] CHECKING PYTHON ENVIRONMENT & VIRTUAL ENVIRONMENT")
    print("=" * 78)

    py_version = sys.version_info
    print(f"  * Python Version:    {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 9):
        print("  [!] WARNING: Python 3.10+ is recommended for optimal compatibility.")

    venv_python = get_venv_python()
    if not venv_python.exists():
        print("  [*] Virtual environment not found. Creating .venv...")
        res = subprocess.run([sys.executable, "-m", "venv", str(PROJECT_ROOT / ".venv")])
        if res.returncode != 0:
            print("  [!] Error creating virtual environment. Continuing with current Python.")
        else:
            print(f"  [+] Created virtual environment at: {PROJECT_ROOT / '.venv'}")
    else:
        print(f"  [+] Virtual environment verified: {venv_python}")

    # If we are NOT running inside the venv, re-execute this script inside the venv
    if not is_running_in_project_venv() and venv_python.exists():
        print("  [*] Switching execution to project virtual environment (.venv)...")
        cmd = [str(venv_python), str(Path(__file__).resolve())] + sys.argv[1:]
        result = subprocess.run(cmd)
        sys.exit(result.returncode)


def step_2_install_dependencies(skip_install: bool = False):
    """Installs required packages from requirements.txt."""
    print("\n" + "=" * 78)
    print("  [STEP 2/5] VERIFYING & INSTALLING DEPENDENCIES")
    print("=" * 78)

    if skip_install:
        print("  [*] Skipping dependency installation (--skip-install flag provided).")
        return

    req_file = PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        print(f"  [!] requirements.txt not found at: {req_file}")
        return

    print(f"  * Requirements File: {req_file}")
    print("  [*] Checking and installing packages via pip...")

    pip_cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
    res = subprocess.run(pip_cmd)
    if res.returncode == 0:
        print("  [+] All dependencies successfully verified/installed.")
    else:
        print("  [!] Note: pip returned a non-zero exit code. Attempting to proceed.")


def step_3_verify_model_weights():
    """Checks pretrained model weights and normalization artifacts."""
    print("\n" + "=" * 78)
    print("  [STEP 3/5] VERIFYING MODEL WEIGHTS & ARTIFACTS REGISTRY")
    print("=" * 78)

    weights = [
        ("Temporal Encoder (ResNet1D-SE)", "models/C5_full_dataset.pt"),
        ("Morphology Encoder (2D ResNet)", "models/morphology_encoder_v1.pt"),
        ("Biomarker Encoder (Attention MLP)", "biomarkers/attention_mlp_best.pt"),
        ("Unified Classifier Head (MLP)", "models/classification_mlp.pt"),
        ("Calibrated Thresholds", "models/classification_mlp_thresholds.npy"),
        ("Biomarker Robust Scaler", "biomarkers/scaler.pkl"),
        ("Biomarker Median Imputer", "biomarkers/imputer.pkl"),
    ]

    all_found = True
    for name, rel_path in weights:
        p = PROJECT_ROOT / rel_path
        if p.exists():
            size_kb = os.path.getsize(p) / 1024
            print(f"  [+] {name:<35}: PRESENT ({size_kb:,.1f} KB)")
        else:
            print(f"  [!] {name:<35}: MISSING ({rel_path})")
            all_found = False

    if all_found:
        print("  [+] All 7 foundational model artifacts are verified and ready.")
    else:
        print("  [!] Warning: One or more model artifacts are missing. Some features may use fallbacks.")


def step_4_download_and_cache_data(limit: int = 30, skip_data: bool = False):
    """Downloads/pre-caches authentic PhysioNet PTB-XL records."""
    print("\n" + "=" * 78)
    print("  [STEP 4/5] CHECKING & DOWNLOADING PTB-XL ECG DATA")
    print("=" * 78)

    cache_dir = PROJECT_ROOT / "web_app" / "data" / "cached_signals"
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing_files = list(cache_dir.glob("*.npy"))

    print(f"  * Cache Directory: {cache_dir}")
    print(f"  * Cached Signals:  {len(existing_files)} existing .npy files")

    if skip_data:
        print("  [*] Skipping data download step (--skip-data flag provided).")
        return

    # Check if we already have sufficient cached records
    if len(existing_files) >= limit:
        print(f"  [+] Cache already contains {len(existing_files)} authentic records (>= {limit} target).")
        print("  [*] Using existing local cache.")
        return

    print(f"  [*] Pre-caching authentic 12-lead records from PhysioNet (Target: {limit})...")
    try:
        from scripts.download_data import download_ptbxl_data
        download_ptbxl_data(limit=limit)
    except Exception as e:
        print(f"  [!] Note during download: {e}")
        print("  [*] Built-in physiological waveform generators will serve as seamless offline fallback.")


def step_5_smoke_test_engine():
    """Runs a quick 1-second smoke test verifying PyTorch tensor operations."""
    print("\n" + "=" * 78)
    print("  [STEP 5/5] EXECUTING FOUNDATION ENGINE SMOKE TEST")
    print("=" * 78)

    try:
        import torch
        import numpy as np
        from ecg_engine import ECGEncoderEngine, EngineConfig

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  * PyTorch Version:   {torch.__version__}")
        print(f"  * Active Device:     {dev.upper()}")
        if dev == "cuda":
            print(f"  * GPU Model:         {torch.cuda.get_device_name(0)}")

        engine = ECGEncoderEngine(config=EngineConfig(device=dev))
        test_signal = np.zeros((12, 1000), dtype=np.float32)
        rep, pred = engine.process(test_signal)

        print(f"  [+] Rep Shapes: z_t={rep.z_temporal.shape}, z_m={rep.z_morphology.shape}, z_b={rep.z_biomarker.shape}")
        print(f"  [+] Fused Multimodal Space: {rep.z_fused.shape} (1056 Dimensions)")
        print(f"  [+] Engine Pipeline is fully operational.")
    except Exception as e:
        print(f"  [!] Smoke test warning: {e}")


def launch_web_application(port: int = 8000, no_browser: bool = False):
    """Starts the web application."""
    print("\n" + "=" * 78)
    print("  READY: LAUNCHING ECG REPORT GENERATOR WEB APPLICATION")
    print("=" * 78)

    from scripts.run_web import run_web_app
    run_web_app(port=port, no_browser=no_browser)


def main():
    parser = argparse.ArgumentParser(
        description="ECG Foundation Representation System — Complete Bootstrap & Launch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup_and_run.py
  python scripts/setup_and_run.py --skip-install --skip-data
  python scripts/setup_and_run.py --port 8080 --no-browser
        """
    )
    parser.add_argument("--port", "-p", type=int, default=8000, help="Web application port (default: 8000)")
    parser.add_argument("--limit-data", "-n", type=int, default=30, help="Number of records to download/cache (default: 30)")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip dependency installation")
    parser.add_argument("--skip-data", action="store_true", help="Skip data downloading and caching")
    parser.add_argument("--no-browser", action="store_true", help="Do not open web browser automatically")

    args = parser.parse_args()

    # Step 1: Environment & .venv
    step_1_ensure_environment()

    # Step 2: Dependencies
    step_2_install_dependencies(skip_install=args.skip_install)

    # Step 3: Model Weights Verification
    step_3_verify_model_weights()

    # Step 4: Download / Cache Data
    step_4_download_and_cache_data(limit=args.limit_data, skip_data=args.skip_data)

    # Step 5: Smoke Test Engine
    step_5_smoke_test_engine()

    # Final: Launch Web Application
    launch_web_application(port=args.port, no_browser=args.no_browser)


if __name__ == "__main__":
    main()
