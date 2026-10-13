@echo off
title ECG Individual Report Generator
chcp 65001 >nul 2>&1
cd /d "%~dp0\.."

if "%~1"=="" (
    set /p SAMPLE_ID="Enter Record Code/ID [Default: REC-01]: "
)

if "%SAMPLE_ID%"=="" set SAMPLE_ID=REC-01

if exist ".venv\Scripts\python.exe" (
    if "%~1"=="" (
        ".venv\Scripts\python.exe" scripts\generate_report.py --sample %SAMPLE_ID% --open --json
    ) else (
        ".venv\Scripts\python.exe" scripts\generate_report.py %*
    )
) else (
    if "%~1"=="" (
        python scripts\generate_report.py --sample %SAMPLE_ID% --open --json
    ) else (
        python scripts\generate_report.py %*
    )
)

pause
