#!/bin/bash

# AI Performance Stack Setup Script
# Automates the setup process for Ubuntu 22.04

set -e

echo "🚀 AI Performance Stack Setup"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on supported OS
check_os() {
    log_info "Checking operating system..."
    
    if [[ ! -f /etc/os-release ]]; then
        log_error "Cannot determine OS version"
        exit 1
    fi
    
    . /etc/os-release
    
    if [[ "$ID" != "ubuntu" ]] || [[ "$VERSION_ID" != "22.04" ]]; then
        log_warn "This script is designed for Ubuntu 22.04. Your OS: $ID $VERSION_ID"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    log_info "OS check completed"
}

# Check for required hardware
check_hardware() {
    log_info "Checking hardware requirements..."
    
    # Check for NVIDIA GPUs
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. Please install NVIDIA drivers first."
        exit 1
    fi
    
    gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n1)
    log_info "Found $gpu_count NVIDIA GPU(s)"
    
    if [[ $gpu_count -lt 2 ]]; then
        log_warn "Only $gpu_count GPU(s) found. Consider modifying docker-compose.yml for single GPU setup."
    fi
    
    # Check available memory
    mem_gb=$(free -g | awk '/^Mem:/{print $2}')
    log_info "Available RAM: ${mem_gb}GB"
    
    if [[ $mem_gb -lt 16 ]]; then
        log_warn "Less than 16GB RAM detected. Some models may not fit in memory."
    fi
    
    log_info "Hardware check completed"
}

# Install Docker
install_docker() {
    if command -v docker &> /dev/null; then
        log_info "Docker already installed: $(docker --version)"
        return 0
    fi
    
    log_info "Installing Docker..."
    
    # Update package list
    sudo apt-get update
    
    # Install prerequisites
    sudo apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    # Add Docker's official GPG key
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    
    # Set up repository
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker Engine
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Add current user to docker group
    sudo usermod -aG docker $USER
    
    log_info "Docker installed successfully"
    log_warn "Please log out and back in for docker group changes to take effect"
}

# Install NVIDIA Container Toolkit
install_nvidia_docker() {
    if docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi &> /dev/null; then
        log_info "NVIDIA Container Toolkit already working"
        return 0
    fi
    
    log_info "Installing NVIDIA Container Toolkit..."
    
    # Add NVIDIA package repository
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    # Install the toolkit
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    
    # Configure Docker daemon
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    log_info "NVIDIA Container Toolkit installed successfully"
}

# Create directory structure
setup_directories() {
    log_info "Setting up directory structure..."
    
    # Create model directories
    mkdir -p models/{deepfacelive,instantid,musicgen,rvc,bark,xtts,demucs,tts}
    
    # Create audio directories (these should already exist but just in case)
    mkdir -p audio/{raw,finished,stems}
    
    # Set permissions
    chmod 755 models audio
    chmod 755 models/* audio/* 2>/dev/null || true
    
    log_info "Directory structure created"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd compose
    
    # Build images
    docker compose build --no-cache
    
    cd ..
    
    log_info "Docker images built successfully"
}

# Test the installation
test_installation() {
    log_info "Testing installation..."
    
    cd compose
    
    # Test GPU access
    log_info "Testing GPU access..."
    if ! docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi; then
        log_error "GPU access test failed"
        return 1
    fi
    
    # Start audio service for testing
    log_info "Starting audio service for testing..."
    docker compose up -d audio
    
    # Wait for service to start
    sleep 30
    
    # Test health endpoint
    if curl -f http://localhost:7861/health; then
        log_info "Audio service health check passed"
    else
        log_error "Audio service health check failed"
        docker compose logs audio
        return 1
    fi
    
    # Stop test services
    docker compose down
    
    cd ..
    
    log_info "Installation test completed successfully"
}

# Print usage instructions
print_usage() {
    log_info "Setup completed successfully! 🎉"
    echo
    echo "Next steps:"
    echo "1. Download AI models to the models/ directory (optional)"
    echo "2. Start services:"
    echo "   - Studio mode (audio only): cd compose && docker compose up -d audio"
    echo "   - Full stream rig: cd compose && docker compose up -d --profile stream"
    echo
    echo "Access points:"
    echo "   - Audio Processing: http://localhost:7861"
    echo "   - Face Swapping: http://localhost:7860 (stream mode only)"
    echo "   - MediaPipe Tracker: http://localhost:7865 (stream mode only)"
    echo
    echo "For more information, see the README.md file."
}

# Main execution
main() {
    echo "This script will install Docker, NVIDIA Container Toolkit, and set up the AI Performance Stack."
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
    
    check_os
    check_hardware
    install_docker
    install_nvidia_docker
    setup_directories
    build_images
    test_installation
    print_usage
}

# Run main function
main "$@"