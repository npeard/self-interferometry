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

@REM echo [Job 2] Training with tcan-config.yaml...
@REM %PYTHON_CMD% ./analysis/models/configs/tcan-config.yaml
@REM if %ERRORLEVEL% NEQ 0 (
@REM     echo ERROR: Job 2 failed with exit code %ERRORLEVEL%
@REM     echo Continue anyway? Press Ctrl+C to stop, or
@REM     pause
@REM )
@REM echo Job 2 completed.
@REM echo.

echo [Job 2] Training with utcn-config.yaml...
%PYTHON_CMD% ./analysis/models/configs/utcn-config.yaml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Job 3 failed with exit code %ERRORLEVEL%
    echo Continue anyway? Press Ctrl+C to stop, or
    pause
)
echo Job 3 completed.
echo.

REM ====================================
REM End of training jobs
REM ====================================

echo ========================================
echo All training jobs completed!
echo ========================================
pause
