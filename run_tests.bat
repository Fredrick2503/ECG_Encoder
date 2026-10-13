@echo off
title ECG Foundation System — Unit & Integration Tests
chcp 65001 >nul 2>&1
echo =======================================================================
echo   Running ECG Foundation Engine Unit Tests...
echo =======================================================================

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo [*] Active Environment: .venv\Scripts\python.exe
    ".venv\Scripts\python.exe" scripts\run_engine.py test %*
) else (
    echo [*] Active Environment: System Python
    python scripts\run_engine.py test %*
)

echo.
pause
