@echo off
REM ============================================================
REM  AI Teacher - Run the test suite
REM  No network access, no API key used, no charges.
REM  NOTE: keep this file pure ASCII (console code page safe).
REM ============================================================
setlocal
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv not found in PATH.
    echo.
    pause
    exit /b 1
)

REM The suite forces the API key to empty and points the base_url at an
REM unroutable address, so a real .env key is never read and never spent.
uv run python -m tests.test_no_network
echo.
pause
endlocal
