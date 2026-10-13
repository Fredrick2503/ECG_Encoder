@echo off
title ECG Foundation System — Unit & Integration Tests
chcp 65001 >nul 2>&1
cd /d "%~dp0\.."

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\run_engine.py test %*
) else (
    python scripts\run_engine.py test %*
)

pause
