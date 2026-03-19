@echo off
REM Training Job Submission Script
REM Created: 2025-12-10

echo ========================================
echo Starting Multi-Config Training Pipeline
echo ========================================
echo.

REM Set the base command
set PYTHON_CMD=python main.py --config

REM ====================================
REM Training Jobs
REM ====================================

echo [Job 1] Training with tcn-config.yaml...
%PYTHON_CMD% ./analysis/models/configs/tcn-config.yaml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Job 1 failed with exit code %ERRORLEVEL%
    echo Continue anyway? Press Ctrl+C to stop, or
    pause
)
echo Job 1 completed.
echo.

echo [Job 2] Training with scnn-config.yaml...
%PYTHON_CMD% ./analysis/models/configs/scnn-config.yaml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Job 2 failed with exit code %ERRORLEVEL%
    echo Continue anyway? Press Ctrl+C to stop, or
    pause
)
echo Job 2 completed.
echo.

REM ====================================
REM End of training jobs
REM ====================================

echo ========================================
echo All training jobs completed!
echo ========================================
pause
