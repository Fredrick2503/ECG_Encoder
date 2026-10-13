@echo off
title ECG Foundation System — Demo & Verification
chcp 65001 >nul 2>&1
cd /d "%~dp0\.."

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\run_engine.py demo %*
) else (
    python scripts\run_engine.py demo %*
)

pause
