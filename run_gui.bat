@echo off
REM ============================================================
REM  AI Teacher - GUI Launcher
REM  Starts the tkinter graphical interface.
REM  NOTE: keep this file pure ASCII. Console code page will
REM        garble non-ASCII characters in .bat files.
REM ============================================================
setlocal
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv not found in PATH.
    echo         Install it from: https://docs.astral.sh/uv/
    echo         Then run this file again.
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

if not exist ".env" (
    echo [WARN] .env not found - API key is not configured.
    echo        The app will run with local rules only.
    echo        Copy .env.example to .env and add your key to enable the model.
    echo.
)

REM Launch the GUI. This call blocks until the window is closed.
uv run ai-teacher-gui
if errorlevel 1 (
    echo.
    echo [ERROR] The GUI exited with an error. Message is above.
    echo         If no window ever appeared, try running this instead:
    echo           uv run python -m ai_teacher.gui
    echo.
    pause
)
endlocal
