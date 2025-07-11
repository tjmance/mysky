@echo off
echo MySky Installer Creation Script
echo ===============================

REM Check if executable exists
if not exist "dist\MySky.exe" (
    echo Error: MySky.exe not found in dist folder
    echo Please run build.bat first to create the executable
    pause
    exit /b 1
)

REM Create installer output directory
if not exist "installer_output" mkdir installer_output

REM Check common Inno Setup installation paths
set ISCC=
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" (
    set ISCC="%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
) else if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" (
    set ISCC="%ProgramFiles%\Inno Setup 6\ISCC.exe"
) else if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    set ISCC="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
) else (
    echo Error: Inno Setup 6 not found!
    echo Please install Inno Setup 6 from: https://jrsoftware.org/isinfo.php
    echo.
    echo After installation, you can compile the installer manually:
    echo 1. Open Inno Setup Compiler
    echo 2. Load mysky_installer.iss
    echo 3. Click "Compile"
    pause
    exit /b 1
)

echo.
echo Found Inno Setup at: %ISCC%
echo.
echo Creating installer...
%ISCC% mysky_installer.iss

if errorlevel 1 (
    echo.
    echo Error: Failed to create installer
    pause
    exit /b 1
)

echo.
echo Installer created successfully!
echo Check the installer_output folder for the setup file.
pause