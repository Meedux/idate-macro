@echo off
title iDate Revival
echo Starting iDate Revival...

:: Activate venv
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment not found!
    echo Run install.bat first.
    pause
    exit /b 1
)

:: Launch (requests admin automatically)
python -m src.gui
