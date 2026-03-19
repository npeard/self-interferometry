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

@REM echo [Job 1] Training with tcn-config.yaml...
@REM %PYTHON_CMD% ./analysis/models/configs/tcn-config.yaml
@REM if %ERRORLEVEL% NEQ 0 (
@REM     echo ERROR: Job 1 failed with exit code %ERRORLEVEL%
@REM     echo Continue anyway? Press Ctrl+C to stop, or
@REM     pause
@REM )
@REM echo Job 1 completed.
@REM echo.

@REM echo [Job 2] Training with scnn-config.yaml...
@REM %PYTHON_CMD% ./analysis/models/configs/scnn-config.yaml
@REM if %ERRORLEVEL% NEQ 0 (
@REM     echo ERROR: Job 2 failed with exit code %ERRORLEVEL%
@REM     echo Continue anyway? Press Ctrl+C to stop, or
@REM     pause
@REM )
@REM echo Job 2 completed.
@REM echo.

@REM echo [Job 3] Training with barland-config.yaml...
@REM %PYTHON_CMD% ./analysis/models/configs/barland-config.yaml
@REM if %ERRORLEVEL% NEQ 0 (
@REM     echo ERROR: Job 3 failed with exit code %ERRORLEVEL%
@REM     echo Continue anyway? Press Ctrl+C to stop, or
@REM     pause
@REM )
@REM echo Job 3 completed.
@REM echo.

@REM echo [Job 4] Training with lstm-config.yaml...
@REM %PYTHON_CMD% ./analysis/models/configs/lstm-config.yaml
@REM if %ERRORLEVEL% NEQ 0 (
@REM     echo ERROR: Job 4 failed with exit code %ERRORLEVEL%
@REM     echo Continue anyway? Press Ctrl+C to stop, or
@REM     pause
@REM )
@REM echo Job 4 completed.
@REM echo.

echo [Job 5] Training with tcan-config.yaml...
%PYTHON_CMD% ./analysis/models/configs/tcan-config.yaml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Job 5 failed with exit code %ERRORLEVEL%
    echo Continue anyway? Press Ctrl+C to stop, or
    pause
)
echo Job 5 completed.
echo.

REM ====================================
REM End of training jobs
REM ====================================

echo ========================================
echo All training jobs completed!
echo ========================================
pause
