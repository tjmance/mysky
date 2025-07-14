FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=1

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
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libasound2-dev \
    libpulse-dev \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for some audio processing tools
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Create working directory
WORKDIR /app

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install core audio processing dependencies
RUN pip3 install --no-cache-dir \
    numpy==1.24.3 \
    scipy \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    pyaudio==0.2.13 \
    pydub==0.25.1 \
    gradio==3.50.0 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    websockets==11.0.3

# Install TTS and voice processing libraries
RUN pip3 install --no-cache-dir \
    TTS==0.22.0 \
    bark-voice-cloning \
    xtts-v2 \
    tortoise-tts \
    coqui-ai-tts

# Install MusicGen and audio generation
RUN pip3 install --no-cache-dir \
    audiocraft==1.1.0 \
    musicgen \
    transformers==4.35.0 \
    accelerate==0.24.1 \
    diffusers==0.21.4

# Install RVC (Retrieval-based Voice Conversion)
RUN pip3 install --no-cache-dir \
    fairseq==0.12.2 \
    praat-parselmouth==0.4.3 \
    pyworld==0.3.4 \
    faiss-gpu==1.7.4

# Install Demucs for source separation
RUN pip3 install --no-cache-dir \
    demucs==4.0.1 \
    julius==0.2.7

# Install additional ML libraries
RUN pip3 install --no-cache-dir \
    huggingface-hub==0.17.3 \
    safetensors==0.4.0 \
    omegaconf==2.3.0 \
    hydra-core==1.3.2

# Clone TTS-WebUI
RUN git clone https://github.com/rsxdalv/tts-generation-webui.git /app/tts-webui

# Clone RVC-WebUI
RUN git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git /app/rvc-webui

# Set up model directories
RUN mkdir -p /app/models/{tts,rvc,musicgen,demucs,bark,xtts} \
    /app/audio/{input,output,stems} \
    /app/temp

# Copy application files
COPY audio_app.py /app/
COPY requirements.txt /app/

# Install additional requirements if any
RUN if [ -f requirements.txt ]; then pip3 install --no-cache-dir -r requirements.txt; fi

# Expose ports for different services
EXPOSE 7861 7862 7863

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES=1
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV HF_HOME=/app/models
ENV TORCH_HOME=/app/models

# Create entrypoint script
RUN echo '#!/bin/bash\n\
export CUDA_VISIBLE_DEVICES=1\n\
export HF_HOME=/app/models\n\
export TORCH_HOME=/app/models\n\
cd /app\n\
python3 audio_app.py "$@"' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--host", "0.0.0.0", "--port", "7861"]