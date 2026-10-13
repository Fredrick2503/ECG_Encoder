"""
ECG Clinical Diagnostic Report Generator — Web Application CLI Runner
======================================================================
Standalone launcher for the Black & White Professional Report Generator Web App.

Features:
- Standard 12-lead diagnostic tracing (4x3 layout + continuous Lead II rhythm strip)
- High-contrast clinical medical stationery design
- Calibrated 5-class superclass prediction (NORM, MI, STTC, CD, HYP)
- 24 physiological biomarkers table with normal reference ranges
- One-click A4 / Letter PDF export and medical report printing
- Multimodal Latent Space Explorer & 3D TSNE/PCA Projections

Usage:
    # Default launch (port 8000, auto-opens browser)
    python scripts/run_web.py

    # Specify custom port
    python scripts/run_web.py --port 8080

    # Headless mode (no browser window)
    python scripts/run_web.py --no-browser
"""

from __future__ import annotations
import sys
import os
import time
import socket
import argparse
import webbrowser
from pathlib import Path
from http.server import HTTPServer

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

from web_app.server import ECGWebAppHandler, STATIC_DIR


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Checks if a TCP port is occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def find_available_port(preferred_port: int = 8000, max_attempts: int = 25) -> int:
    """Returns the preferred port if available, or the next free sequential port."""
    for p in range(preferred_port, preferred_port + max_attempts):
        if not is_port_in_use(p):
            return p
    return preferred_port


def print_web_banner(port: int, url: str, host: str):
    print("\n" + "=" * 78)
    print("  [ECG CORE LAB] CLINICAL DIAGNOSTIC REPORT GENERATOR — WEB APPLICATION")
    print("  Multimodal Foundation Representation System (Tri-Modal 1056-D Architecture)")
    print("=" * 78)
    print(f"  * Status:          ONLINE & ACTIVE")
    print(f"  * Local Web URL:   {url}")
    print(f"  * Network Bind:    http://{host}:{port}")
    print(f"  * Static UI Path:  {STATIC_DIR}")
    print(f"  * Interface Theme: Monochrome Clinical Stationery (High-Contrast Print)")
    print(f"  * Key Features:    - 12-Lead Diagnostic Waveform Tracing (4x3 + Lead II Strip)")
    print(f"                     - Calibrated Multi-Label Diagnosis (NORM, MI, STTC, CD, HYP)")
    print(f"                     - 24 Electrophysiological Biomarkers Table & Reference Ranges")
    print(f"                     - Real-Time PTB-XL Sample Catalog Browser (600 Records)")
    print(f"                     - One-Click Official Clinical Report Print (A4 / Letter PDF)")
    print("=" * 78)
    print(f"  --> Opening default browser at: {url}")
    print("  --> Press CTRL+C at any time in this console to terminate server.\n")


def run_web_app(port: int = 8000, host: str = "0.0.0.0", no_browser: bool = False):
    actual_port = find_available_port(port)
    url = f"http://localhost:{actual_port}"

    server_address = (host, actual_port)
    try:
        httpd = HTTPServer(server_address, ECGWebAppHandler)
    except OSError as e:
        print(f"[!] Could not bind to {host}:{actual_port}: {e}")
        sys.exit(1)

    print_web_banner(actual_port, url, host)

    if not no_browser:
        def open_browser():
            time.sleep(0.7)
            webbrowser.open(url)

        import threading
        threading.Thread(target=open_browser, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Received shutdown signal (Ctrl+C). Stopping ECG Web App Server.")
        httpd.server_close()
        print("[*] Web server stopped cleanly. Goodbye.\n")


def main():
    parser = argparse.ArgumentParser(
        description="ECG Foundation Representation System — Web Application Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_web.py
  python scripts/run_web.py --port 8080
  python scripts/run_web.py --no-browser
        """
    )
    parser.add_argument("--port", "-p", type=int, default=8000, help="Preferred port (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open web browser automatically")

    args = parser.parse_args()
    run_web_app(port=args.port, host=args.host, no_browser=args.no_browser)


if __name__ == "__main__":
    main()
