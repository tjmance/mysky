FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install Python dependencies for face swapping
RUN pip3 install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install --no-cache-dir \
    opencv-python==4.8.1.78 \
    mediapipe==0.10.7 \
    numpy==1.24.3 \
    pillow==10.0.1 \
    insightface==0.7.3 \
    onnxruntime-gpu==1.16.1 \
    diffusers==0.21.4 \
    transformers==4.35.0 \
    accelerate==0.24.1 \
    xformers==0.0.22 \
    controlnet-aux==0.0.6 \
    face-alignment==1.4.1 \
    scipy \
    scikit-image \
    gradio==3.50.0 \
    fastapi==0.104.1 \
    uvicorn==0.24.0

# Clone DeepFaceLive
RUN git clone https://github.com/iperov/DeepFaceLive.git /app/DeepFaceLive

# Clone InstantID
RUN git clone https://github.com/InstantID/InstantID.git /app/InstantID

# Set up MediaPipe models
RUN mkdir -p /app/mediapipe_models && \
    wget -O /app/mediapipe_models/face_detection_short_range.tflite \
    https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

# Create necessary directories
RUN mkdir -p /app/models /app/temp /app/output

# Copy application files
COPY faceswap_app.py /app/
COPY requirements.txt /app/

# Install additional requirements if any
RUN if [ -f requirements.txt ]; then pip3 install --no-cache-dir -r requirements.txt; fi

# Expose port for web interface
EXPOSE 7860

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES=0
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Create entrypoint script
RUN echo '#!/bin/bash\n\
export CUDA_VISIBLE_DEVICES=0\n\
cd /app\n\
python3 faceswap_app.py "$@"' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--host", "0.0.0.0", "--port", "7860"]