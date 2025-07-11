# MySky - Windows Desktop Application

A simple desktop application for Windows with a professional installer.

## Features

- Simple GUI application built with Python and Tkinter
- Professional Windows installer
- Desktop shortcut creation
- Clean uninstallation

## Project Structure

```
mysky/
├── mysky.py                 # Main application
├── build_windows.py         # Build script for creating executable
├── mysky_installer.iss      # Inno Setup script for installer
├── build.bat               # Windows batch script to build exe
├── create_installer.bat    # Windows batch script to create installer
├── create_icon.py          # Script to generate application icon
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Prerequisites

1. **Python 3.x** - Download from [python.org](https://www.python.org/downloads/)
2. **Inno Setup 6** - Download from [jrsoftware.org](https://jrsoftware.org/isdl.php)

## Building the Windows Installer

### Step 1: Create the Application Icon

```bash
pip install Pillow
python create_icon.py
```

This creates `mysky.ico` for the application.

### Step 2: Build the Executable

Run the build script:
```bash
build.bat
```

Or manually:
```bash
pip install -r requirements.txt
python build_windows.py
```

This will:
- Install PyInstaller
- Create a standalone `MySky.exe` in the `dist/` folder

### Step 3: Create the Installer

Run the installer creation script:
```bash
create_installer.bat
```

This will:
- Check for the executable in `dist/`
- Use Inno Setup to create the installer
- Output the installer to `installer_output/MySky_Setup_1.0.0.exe`

## Manual Build Process

If you prefer to build manually:

1. **Install dependencies:**
   ```bash
   pip install pyinstaller
   ```

2. **Build executable:**
   ```bash
   pyinstaller --name=MySky --onefile --windowed --icon=mysky.ico mysky.py
   ```

3. **Create installer:**
   - Open Inno Setup Compiler
   - Load `mysky_installer.iss`
   - Click "Compile"

## Distribution

The final installer will be in the `installer_output/` folder:
- `MySky_Setup_1.0.0.exe` - Ready for distribution

Users can run this installer to:
- Install MySky to Program Files
- Create Start Menu shortcuts
- Optionally create Desktop shortcuts
- Properly register for Windows uninstallation

## Development

To test the application without building:
```bash
python mysky.py
```

## Customization

### Application Details
Edit these in `mysky.py`:
- Application title and size
- GUI elements and functionality
- Version information

### Installer Details
Edit these in `mysky_installer.iss`:
- Company/publisher information
- Installation directory
- File associations
- Registry entries

## Troubleshooting

1. **"Python not found"** - Ensure Python is in your PATH
2. **"Inno Setup not found"** - Install Inno Setup 6 or compile manually
3. **Icon issues** - Run `create_icon.py` to generate the icon file
4. **Build errors** - Check that all dependencies are installed

## License

This is a sample project for demonstration purposes.