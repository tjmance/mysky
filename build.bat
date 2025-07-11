@echo off
echo MySky Windows Build Script
echo ==========================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.x and add it to PATH
    pause
    exit /b 1
)

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt

REM Run the build script
echo.
echo Building MySky executable...
python build_windows.py

echo.
echo Build process completed!
pause