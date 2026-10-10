@echo off
REM ============================================================
REM  AI Teacher - CLI Launcher
REM  Starts the terminal monitor. Press Ctrl+C to stop.
REM  NOTE: keep this file pure ASCII (console code page safe).
REM ============================================================
setlocal
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv not found in PATH.
    echo         Install it from: https://docs.astral.sh/uv/
    echo.
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [INFO] First run detected. Creating virtual environment...
    uv sync
    if errorlevel 1 (
        echo [ERROR] uv sync failed. Check your network and retry.
        pause
        exit /b 1
    )
)

uv run ai-teacher start %*
if errorlevel 1 (
    echo.
    echo [ERROR] Exited with an error. Scroll up for details.
    pause
)
endlocal
