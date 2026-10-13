@echo off
title ECG Foundation Engine — Verification Demo & Diagnostics
chcp 65001 >nul 2>&1
echo =======================================================================
echo   Launching ECG Foundation Representation Engine...
echo =======================================================================

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo [*] Active Environment: .venv\Scripts\python.exe
    ".venv\Scripts\python.exe" scripts\run_engine.py %*
) else (
    echo [*] Active Environment: System Python
    python scripts\run_engine.py %*
)

echo.
pause
