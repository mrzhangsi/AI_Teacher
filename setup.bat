@echo off
REM ============================================================
REM  AI Teacher - First-time setup
REM  Creates .env from the template and runs a self check.
REM  NOTE: keep this file pure ASCII (console code page safe).
REM ============================================================
setlocal
cd /d "%~dp0"

echo ==========================================
echo  AI Teacher - Setup
echo ==========================================
echo.

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv not found in PATH.
    echo         Install it from: https://docs.astral.sh/uv/
    echo.
    pause
    exit /b 1
)

echo [1/3] Installing dependencies...
uv sync
if errorlevel 1 (
    echo [ERROR] uv sync failed. Check your network and retry.
    pause
    exit /b 1
)
echo       done.
echo.

echo [2/3] Preparing .env...
if exist ".env" (
    echo       .env already exists, skipping.
) else (
    copy ".env.example" ".env" >nul
    echo       .env created. Open it and set AI_TEACHER_API_KEY.
)
echo.

echo [3/3] Running self check...
echo.
uv run python -m tests.test_e2e
echo.

echo ==========================================
echo  Setup finished.
echo.
echo  Next steps:
echo    1. Edit .env and paste your DeepSeek API key
echo    2. Double-click run_gui.bat   (graphical)
echo       or run_cli.bat            (terminal)
echo ==========================================
echo.
pause
endlocal
