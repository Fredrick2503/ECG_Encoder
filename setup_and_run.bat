@echo off
title ECG Foundation System — Setup, Download Data & Launch Web App
chcp 65001 >nul 2>&1

echo =======================================================================
echo   ECG FOUNDATION REPRESENTATION SYSTEM — ALL-IN-ONE BOOTSTRAP
echo   Setup Environment • Download/Cache Data • Launch Web Application
echo =======================================================================

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo [*] Active Environment: .venv\Scripts\python.exe
    ".venv\Scripts\python.exe" scripts\setup_and_run.py %*
) else (
    echo [*] Active Environment: System Python (will initialize .venv)
    python scripts\setup_and_run.py %*
)

if errorlevel 1 (
    echo.
    echo [!] Bootstrap encountered an issue.
    pause
)
