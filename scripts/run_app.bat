@echo off
title ECG Foundation System — Master Launcher
chcp 65001 >nul 2>&1
cd /d "%~dp0\.."

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\run_app.py %*
) else (
    python scripts\run_app.py %*
)

pause
