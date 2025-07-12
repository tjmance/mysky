THIS SHOULD BE A LINTER ERROR#!/usr/bin/env python3
"""
Setup script for AI Video Generation Studio
Installs dependencies and configures the environment
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("=" * 60)
    print("🎬 AI Video Generation Studio Setup")
    print("   Powered by SkyReels v2")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_system_requirements():
    """Check system requirements"""
    print("\n🔍 Checking system requirements...")
    
    # Check OS
    system = platform.system()
    if system != "Linux":
        print(f"⚠️ This system is optimized for Linux, detected: {system}")
    else:
        print(f"✅ Operating System: {system}")
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / 1024**3
        if memory_gb < 16:
            print(f"⚠️ Low RAM detected: {memory_gb:.1f} GB (16GB+ recommended)")
        else:
            print(f"✅ RAM: {memory_gb:.1f} GB")
    except ImportError:
        print("⚠️ Could not check RAM (psutil not installed)")
    
    # Check disk space
    try:
        disk_free_gb = psutil.disk_usage('/').free / 1024**3
        if disk_free_gb < 50:
            print(f"⚠️ Low disk space: {disk_free_gb:.1f} GB (50GB+ recommended)")
        else:
            print(f"✅ Free disk space: {disk_free_gb:.1f} GB")
    except:
        print("⚠️ Could not check disk space")

def install_pytorch():
    """Install PyTorch with CUDA support"""
    print("\n📦 Installing PyTorch with CUDA support...")
    
    try:
        # Install PyTorch with CUDA 12.1 support
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ PyTorch installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch: {e}")
        print("Falling back to CPU-only PyTorch...")
        
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"], check=True)
            print("✅ PyTorch (CPU) installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install PyTorch")
            return False

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def install_system_dependencies():
    """Install system dependencies (ffmpeg, etc.)"""
    print("\n🔧 Installing system dependencies...")
    
    system = platform.system()
    
    if system == "Linux":
        # Try different package managers
        package_managers = [
            (["sudo", "apt", "update"], ["sudo", "apt", "install", "-y", "ffmpeg", "git"]),
            (["sudo", "yum", "update"], ["sudo", "yum", "install", "-y", "ffmpeg", "git"]),
            (["sudo", "pacman", "-Sy"], ["sudo", "pacman", "-S", "--noconfirm", "ffmpeg", "git"])
        ]
        
        for update_cmd, install_cmd in package_managers:
            try:
                subprocess.run(update_cmd, check=True, capture_output=True)
                subprocess.run(install_cmd, check=True, capture_output=True)
                print("✅ System dependencies installed")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        print("⚠️ Could not install system dependencies automatically")
        print("Please install ffmpeg and git manually:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg git")
        print("  CentOS/RHEL: sudo yum install ffmpeg git")
        print("  Arch: sudo pacman -S ffmpeg git")
        
    else:
        print(f"⚠️ System dependency installation not supported for {system}")
        print("Please install ffmpeg and git manually")
    
    return False

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    directories = [
        "outputs",
        "models", 
        "temp",
        "uploads",
        "outputs/metadata"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def download_skyreels():
    """Download SkyReels repository"""
    print("\n📥 Setting up SkyReels v2...")
    
    skyreels_path = Path("models/SkyReels")
    
    if skyreels_path.exists():
        print("✅ SkyReels directory already exists")
        return True
    
    try:
        subprocess.run([
            "git", "clone", 
            "https://github.com/SkyworkAI/SkyReels.git",
            str(skyreels_path)
        ], check=True, capture_output=True)
        
        print("✅ SkyReels repository downloaded")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Failed to download SkyReels: {e}")
        print("SkyReels will be set up automatically on first run")
        return False

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    # Test PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ CUDA GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("⚠️ CUDA not available - using CPU mode")
            
    except ImportError:
        print("❌ PyTorch import failed")
        return False
    
    # Test other dependencies
    dependencies = [
        ("opencv-python", "cv2"),
        ("Pillow", "PIL"),
        ("numpy", "numpy"),
        ("streamlit", "streamlit")
    ]
    
    for pkg_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"✅ {pkg_name}")
        except ImportError:
            print(f"❌ {pkg_name} import failed")
            return False
    
    # Test ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✅ ffmpeg")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ ffmpeg not available (video concatenation may not work)")
    
    return True

def create_launch_script():
    """Create launch script"""
    print("\n📝 Creating launch script...")
    
    launch_script = """#!/bin/bash
# AI Video Generation Studio Launch Script

echo "🎬 Starting AI Video Generation Studio..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Launch Streamlit app
echo "Launching Streamlit app..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

echo "✅ AI Video Generation Studio started!"
echo "Open your browser to: http://localhost:8501"
"""
    
    with open("launch.sh", "w") as f:
        f.write(launch_script)
    
    # Make executable
    os.chmod("launch.sh", 0o755)
    print("✅ Created launch.sh")

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check system requirements
    check_system_requirements()
    
    # Setup directories
    setup_directories()
    
    # Install system dependencies
    install_system_dependencies()
    
    # Install PyTorch
    if not install_pytorch():
        print("❌ PyTorch installation failed")
        sys.exit(1)
    
    # Install Python dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed")
        sys.exit(1)
    
    # Download SkyReels
    download_skyreels()
    
    # Test installation
    if not test_installation():
        print("❌ Installation test failed")
        sys.exit(1)
    
    # Create launch script
    create_launch_script()
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("")
    print("To start the AI Video Generation Studio:")
    print("  ./launch.sh")
    print("")
    print("Or manually:")
    print("  streamlit run app.py")
    print("")
    print("Then open: http://localhost:8501")
    print("=" * 60)

if __name__ == "__main__":
    main()