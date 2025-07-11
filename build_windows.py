#!/usr/bin/env python3
"""
Build script for creating Windows executable
"""

import os
import sys
import shutil
import subprocess


def clean_build():
    """Clean previous build artifacts"""
    dirs_to_remove = ['build', 'dist', '__pycache__']
    files_to_remove = ['mysky.spec']
    
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Removed {dir_name}")
    
    for file_name in files_to_remove:
        if os.path.exists(file_name):
            os.remove(file_name)
            print(f"Removed {file_name}")


def build_executable():
    """Build the Windows executable using PyInstaller"""
    print("Building MySky executable...")
    
    # PyInstaller command with options
    command = [
        'pyinstaller',
        '--name=MySky',
        '--onefile',
        '--windowed',
        '--icon=mysky.ico',
        '--add-data=mysky.ico;.',
        '--distpath=dist',
        '--clean',
        'mysky.py'
    ]
    
    # Run PyInstaller
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Build successful!")
        print(f"Executable created at: dist/MySky.exe")
    else:
        print("Build failed!")
        print(result.stderr)
        sys.exit(1)


def main():
    """Main build process"""
    print("MySky Windows Build Script")
    print("=" * 50)
    
    # Clean previous builds
    clean_build()
    
    # Create icon if it doesn't exist
    if not os.path.exists('mysky.ico'):
        print("Warning: mysky.ico not found. Building without icon.")
        # Remove icon option from build
        
    # Build executable
    build_executable()
    
    print("\nBuild complete!")
    print("Next steps:")
    print("1. Test the executable in dist/MySky.exe")
    print("2. Run create_installer.bat to create the installer")


if __name__ == "__main__":
    main()