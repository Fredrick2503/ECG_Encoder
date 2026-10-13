"""
ECG Clinical Diagnostic Report Generator — Web Application Launcher
=====================================================================
Runs the Black & White Professional Report Generator Web Application,
serving the multimodal foundation representation model, real PTB-XL records,
and automated clinical report export.
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

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web_app.server import ECGWebAppHandler, STATIC_DIR


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Checks if a local TCP port is already occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def find_available_port(preferred_port: int = 8000, max_attempts: int = 20) -> int:
    """Returns the preferred port if free, or the first available sequential port."""
    for p in range(preferred_port, preferred_port + max_attempts):
        if not is_port_in_use(p):
            return p
    return preferred_port


def print_banner(port: int, url: str):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print("\n" + "=" * 76)
    print("  [ECG CORE LAB] CLINICAL DIAGNOSTIC REPORT GENERATOR - WEB APP")
    print("  Multimodal Foundation Representation System (Tri-Modal 1056-D Architecture)")
    print("=" * 76)
    print(f"  * Status:          ONLINE & READY")
    print(f"  * Local URL:       {url}")
    print(f"  * Static Assets:   {STATIC_DIR}")
    print(f"  * Theme:           Black & White Medical Stationery / High-Contrast Print")
    print(f"  * Features:        12-Lead Diagnostic Tracing (4x3 + Lead II Strip)")
    print(f"                     Calibrated Superclass Classification (NORM, MI, STTC, CD, HYP)")
    print(f"                     24 Electrophysiological Biomarkers & Intervals Table")
    print(f"                     One-Click Official Medical Report Print (A4 / PDF)")
    print("=" * 76)
    print(f"  --> Opening default browser at: {url}")
    print("  --> Press CTRL+C at any time in this console to stop the server.\n")


def run(port: int = 8000, no_browser: bool = False):
    actual_port = find_available_port(port)
    url = f"http://localhost:{actual_port}"

    server_address = ("0.0.0.0", actual_port)
    try:
        httpd = HTTPServer(server_address, ECGWebAppHandler)
    except OSError as e:
        print(f"[!] Could not bind to port {actual_port}: {e}")
        sys.exit(1)

    print_banner(actual_port, url)

    # Automatically launch user browser
    if not no_browser:
        try:
            # Small delay to ensure server loop is ready
            def open_browser():
                time.sleep(0.6)
                webbrowser.open(url)

            import threading
            threading.Thread(target=open_browser, daemon=True).start()
        except Exception as e:
            print(f"[!] Note on auto-browser: {e}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Received shutdown signal (Ctrl+C). Terminating ECG Report Generator.")
        httpd.server_close()
        print("[*] Server shutdown cleanly. Goodbye.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ECG Clinical Report Generator Web App")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Preferred port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open web browser")
    args = parser.parse_args()

    run(port=args.port, no_browser=args.no_browser)
