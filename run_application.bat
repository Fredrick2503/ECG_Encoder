@echo off
title ECG Foundation Representation System — Master Launcher
chcp 65001 >nul 2>&1
echo =======================================================================
echo   ECG Foundation Representation System — Unified Master Launcher
echo =======================================================================

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo [*] Active Environment: .venv\Scripts\python.exe
    ".venv\Scripts\python.exe" scripts\run_app.py %*
) else (
    echo [*] Active Environment: System Python
    python scripts\run_app.py %*
)

if errorlevel 1 (
    echo.
    echo [!] Process exited with an error.
    pause
)
