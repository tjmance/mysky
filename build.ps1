# MySky Windows Build Script (PowerShell)

Write-Host "MySky Windows Build Script" -ForegroundColor Cyan
Write-Host "==========================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.x from https://python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
pip install Pillow  # For icon creation

# Create icon if it doesn't exist
if (-not (Test-Path "mysky.ico")) {
    Write-Host "`nCreating application icon..." -ForegroundColor Yellow
    python create_icon.py
}

# Build executable
Write-Host "`nBuilding MySky executable..." -ForegroundColor Yellow
python build_windows.py

# Check if build was successful
if (Test-Path "dist\MySky.exe") {
    Write-Host "`nBuild successful!" -ForegroundColor Green
    Write-Host "Executable created at: dist\MySky.exe" -ForegroundColor Green
    
    # Create installer if Inno Setup is available
    $innoSetupPaths = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
    )
    
    $isccPath = $null
    foreach ($path in $innoSetupPaths) {
        if (Test-Path $path) {
            $isccPath = $path
            break
        }
    }
    
    if ($isccPath) {
        Write-Host "`nCreating installer..." -ForegroundColor Yellow
        & $isccPath "mysky_installer.iss"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`nInstaller created successfully!" -ForegroundColor Green
            Write-Host "Check the installer_output folder for the setup file." -ForegroundColor Green
        }
    } else {
        Write-Host "`nInno Setup not found. Skipping installer creation." -ForegroundColor Yellow
        Write-Host "To create installer, install Inno Setup from: https://jrsoftware.org/isdl.php" -ForegroundColor Yellow
    }
} else {
    Write-Host "`nBuild failed!" -ForegroundColor Red
}

Write-Host "`nPress Enter to exit..."
Read-Host