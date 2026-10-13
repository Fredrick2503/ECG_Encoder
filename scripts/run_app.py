"""
ECG Foundation Representation System — Master Application Runner
================================================================
Unified entrypoint to launch the web dashboard, engine pipelines,
individual report generator, and diagnostics.

Usage:
    # Interactive Console Menu (Default)
    python scripts/run_app.py

    # Launch Web Application
    python scripts/run_app.py web [--port 8000]

    # Run ECG Foundation Engine
    python scripts/run_app.py engine [demo | info | test | predict | encode]

    # Generate Individual Diagnostic Report
    python scripts/run_app.py report [--sample REC-21826] [--open]

    # Inspect Weights Registry
    python scripts/run_app.py info

    # Run Unit Tests
    python scripts/run_app.py test
"""

from __future__ import annotations
import sys
import os
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


def interactive_menu():
    """Interactive command-line dashboard."""
    from scripts.run_web import run_web_app
    from scripts.run_engine import run_demo, print_info, run_tests
    from scripts.generate_report import generate_individual_report, list_samples

    while True:
        print("\n" + "=" * 78)
        print("  ECG FOUNDATION REPRESENTATION SYSTEM — MASTER CONTROL PANEL")
        print("  Tri-Modal 1056-D Foundation Architecture & Clinical Decision Support")
        print("=" * 78)
        print("  Please select an operation:")
        print("")
        print("    [1] Launch Web Application (Clinical Report Generator & Visualizer)")
        print("    [2] Run ECG Foundation Engine (Verification Demo & 12-Lead Inference)")
        print("    [3] Generate Individual ECG Clinical Diagnostic Report (HTML + Summary)")
        print("    [4] Inspect Model Architecture & Frozen Weights Registry")
        print("    [5] Run Engine Verification Unit Tests")
        print("    [6] List Available PTB-XL ECG Catalog Records (600 Records)")
        print("    [0] Exit")
        print("-" * 78)

        choice = input("  Enter selection [0-6] (Default: 1): ").strip()
        if not choice:
            choice = "1"

        if choice == "1":
            port_input = input("  Enter port [default: 8000]: ").strip()
            port = int(port_input) if port_input.isdigit() else 8000
            run_web_app(port=port)
            break
        elif choice == "2":
            run_demo()
            input("\nPress Enter to return to main menu...")
        elif choice == "3":
            sample_code = input("  Enter sample code or record ID [e.g. REC-01, REC-21826] (Default: REC-01): ").strip()
            if not sample_code:
                sample_code = "REC-01"
            open_resp = input("  Open HTML report in browser? (Y/n): ").strip().lower()
            open_browser = (open_resp != "n")
            generate_individual_report(sample_id=sample_code, open_browser=open_browser, export_json=True)
            input("\nPress Enter to return to main menu...")
        elif choice == "4":
            print_info()
            input("\nPress Enter to return to main menu...")
        elif choice == "5":
            run_tests()
            input("\nPress Enter to return to main menu...")
        elif choice == "6":
            list_samples()
            input("\nPress Enter to return to main menu...")
        elif choice in ("0", "q", "exit"):
            print("\nExiting ECG Foundation Representation System. Goodbye.\n")
            sys.exit(0)
        else:
            print("[!] Invalid selection. Please enter a number between 0 and 6.")


def main():
    parser = argparse.ArgumentParser(
        description="ECG Foundation Representation System — Master Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_app.py
  python scripts/run_app.py web --port 8000
  python scripts/run_app.py engine demo
  python scripts/run_app.py report --sample REC-01 --open
  python scripts/run_app.py info
  python scripts/run_app.py test
        """
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    # Web Subcommand
    web_p = subparsers.add_parser("web", help="Launch the web application")
    web_p.add_argument("--port", "-p", type=int, default=8000, help="Port to host (default: 8000)")
    web_p.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    web_p.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")

    # Engine Subcommand
    engine_p = subparsers.add_parser("engine", help="Run the foundation engine")
    engine_p.add_argument("mode", choices=["demo", "info", "test", "predict", "encode"], default="demo", nargs="?", help="Engine action")
    engine_p.add_argument("--input", "-i", type=str, help="Input .npy/.csv signal file for predict/encode")
    engine_p.add_argument("--output", "-o", type=str, default="z_fused.npy", help="Output file for encode")
    engine_p.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Compute device")

    # Report Subcommand
    report_p = subparsers.add_parser("report", help="Generate individual ECG clinical report")
    report_p.add_argument("--sample", "-s", type=str, default="REC-01", help="Sample code or record ID (e.g. REC-01)")
    report_p.add_argument("--input", "-i", type=str, default=None, help="Custom .npy/.csv signal file")
    report_p.add_argument("--output", "-o", type=str, default=None, help="Custom HTML report output path")
    report_p.add_argument("--json", action="store_true", default=True, help="Export JSON summary (default: True)")
    report_p.add_argument("--open", action="store_true", default=True, help="Auto-open in browser (default: True)")
    report_p.add_argument("--no-open", dest="open", action="store_false", help="Do not auto-open in browser")
    report_p.add_argument("--list", "-l", action="store_true", help="List preset catalog records")

    # Info Subcommand
    subparsers.add_parser("info", help="Inspect model architecture and weights registry")

    # Test Subcommand
    subparsers.add_parser("test", help="Run engine verification tests")

    args = parser.parse_args()

    # If no subcommand provided, launch interactive menu
    if args.subcommand is None:
        interactive_menu()
        return

    if args.subcommand == "web":
        from scripts.run_web import run_web_app
        run_web_app(port=args.port, host=args.host, no_browser=args.no_browser)
    elif args.subcommand == "engine":
        from scripts.run_engine import run_demo, print_info, run_tests, run_predict, run_encode
        mode = args.mode or "demo"
        if mode == "demo":
            run_demo(device=args.device)
        elif mode == "info":
            print_info()
        elif mode == "test":
            sys.exit(run_tests())
        elif mode == "predict":
            if not args.input:
                print("[!] Error: --input required for predict mode.")
                sys.exit(1)
            run_predict(input_path=args.input, device=args.device)
        elif mode == "encode":
            if not args.input:
                print("[!] Error: --input required for encode mode.")
                sys.exit(1)
            run_encode(input_path=args.input, output_path=args.output, device=args.device)
    elif args.subcommand == "report":
        from scripts.generate_report import generate_individual_report, list_samples
        if args.list:
            list_samples()
        else:
            generate_individual_report(
                sample_id=args.sample,
                input_file=args.input,
                output_html=args.output,
                export_json=args.json,
                open_browser=args.open
            )
    elif args.subcommand == "info":
        from scripts.run_engine import print_info
        print_info()
    elif args.subcommand == "test":
        from scripts.run_engine import run_tests
        sys.exit(run_tests())


if __name__ == "__main__":
    main()
