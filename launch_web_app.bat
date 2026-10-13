@echo off
title ECG Clinical Diagnostic Report Generator Web App
chcp 65001 >nul 2>&1
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\run_web.py %*
) else (
    python scripts\run_web.py %*
)

pause
