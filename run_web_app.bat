@echo off
title ECG Clinical Diagnostic Report Generator — Web Application
chcp 65001 >nul 2>&1
echo =======================================================================
echo   Launching ECG Clinical Diagnostic Report Generator Web App...
echo =======================================================================

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo [*] Active Environment: .venv\Scripts\python.exe
    ".venv\Scripts\python.exe" scripts\run_web.py %*
) else (
    echo [*] Active Environment: System Python
    python scripts\run_web.py %*
)

echo.
pause
