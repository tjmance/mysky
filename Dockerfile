# AI Video Generation Studio Docker Container
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-advanced.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Pre-install the heavyweight Hugging Face stack with CUDA wheels first to avoid resolver back-tracking
RUN pip install --no-cache-dir \
        torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
        diffusers==0.28.0 transformers==4.41.2 accelerate==0.29.3 \
        --extra-index-url https://download.pytorch.org/whl/cu121
# Install the remaining advanced dependencies without re-resolving the HF stack
RUN pip install --no-cache-dir --no-deps -r requirements-advanced.txt

# (The HF stack is already installed above, so no separate PyTorch step is necessary)

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p outputs models temp uploads logs

# Set permissions
RUN chmod +x launch.sh setup.py test_system.py

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]