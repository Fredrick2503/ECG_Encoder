@echo off
title ECG Foundation System — Setup, Download Data & Launch Web App
chcp 65001 >nul 2>&1
cd /d "%~dp0\.."

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\setup_and_run.py %*
) else (
    python scripts\setup_and_run.py %*
)

pause
