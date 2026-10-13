@echo off
title ECG Clinical Diagnostic Report Generator — Individual Record
chcp 65001 >nul 2>&1
echo =======================================================================
echo   ECG Foundation System — Individual Clinical Report Generator
echo =======================================================================

cd /d "%~dp0"

if "%~1"=="" (
    echo.
    echo Examples of preset samples:
    echo   REC-01    (Sinus Rhythm)
    echo   REC-21826 (Myocardial Infarction / STTC / Hypertrophy)
    echo   REC-2929  (Normal Variant)
    echo.
    set /p SAMPLE_ID="Enter Record Code/ID [Default: REC-01]: "
)

if "%SAMPLE_ID%"=="" set SAMPLE_ID=REC-01

if exist ".venv\Scripts\python.exe" (
    echo [*] Active Environment: .venv\Scripts\python.exe
    if "%~1"=="" (
        ".venv\Scripts\python.exe" scripts\generate_report.py --sample %SAMPLE_ID% --open --json
    ) else (
        ".venv\Scripts\python.exe" scripts\generate_report.py %*
    )
) else (
    echo [*] Active Environment: System Python
    if "%~1"=="" (
        python scripts\generate_report.py --sample %SAMPLE_ID% --open --json
    ) else (
        python scripts\generate_report.py %*
    )
)

echo.
pause
