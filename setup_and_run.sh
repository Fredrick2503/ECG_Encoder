#!/usr/bin/env bash
# ECG Foundation System — Complete Bootstrap & Web App Launcher
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================="
echo "  ECG FOUNDATION REPRESENTATION SYSTEM — ALL-IN-ONE BOOTSTRAP"
echo "  Setup Environment • Download/Cache Data • Launch Web Application"
echo "======================================================================="

if [ -f ".venv/bin/python" ]; then
    echo "[*] Active Environment: .venv/bin/python"
    ".venv/bin/python" scripts/setup_and_run.py "$@"
else
    echo "[*] Active Environment: System Python (will initialize .venv)"
    python3 scripts/setup_and_run.py "$@"
fi
