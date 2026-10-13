@echo off
title ECG Clinical Diagnostic Report Generator
echo =======================================================================
echo   Launching ECG Clinical Diagnostic Report Generator Web App...
echo =======================================================================

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo [*] Using project virtual environment: .venv\Scripts\python.exe
    ".venv\Scripts\python.exe" run_report_app.py %*
) else (
    echo [*] Using system python...
    python run_report_app.py %*
)

pause
