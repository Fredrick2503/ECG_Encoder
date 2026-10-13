@echo off
title ECG Foundation Engine Runner
chcp 65001 >nul 2>&1
cd /d "%~dp0\.."

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\run_engine.py %*
) else (
    python scripts\run_engine.py %*
)

pause
