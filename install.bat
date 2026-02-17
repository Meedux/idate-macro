@echo off
title iDate Revival - Installation
echo ═══════════════════════════════════════════
echo        iDate Revival - Installation
echo ═══════════════════════════════════════════
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed!
    echo Download Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo       Virtual environment created.
) else (
    echo       Virtual environment already exists.
)

echo.
echo [2/3] Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo [3/3] Installing dependencies...
pip install -e . --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies!
    pause
    exit /b 1
)

echo.
echo ═══════════════════════════════════════════
echo    Installation complete!
echo.
echo    Run "run.bat" to start the program.
echo ═══════════════════════════════════════════
echo.
pause
