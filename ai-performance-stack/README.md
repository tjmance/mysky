# AI Performance Stack

*One repo, two lean images, zero manual installs.*

A Docker-based monorepo for real-time AI performance tools including face swapping, voice changing, and audio generation. Stream with real-time face/voice swap today; flip off the faceswap containers tomorrow and spin up MusicGen or YuE for full-length tracks—all without touching the host OS or juggling dependencies.

## 🏗️ Architecture

### Repository Structure
```
ai-performance-stack/
├── docker/
│   ├── faceswap/          # DeepFaceLive + InstantID + MediaPipe
│   │   ├── Dockerfile
│   │   └── faceswap_app.py
│   └── audio/             # TTS-WebUI bundle (Bark, XTTS, MusicGen, RVC, Demucs)
│       ├── Dockerfile
│       └── audio_app.py
├── compose/
│   └── docker-compose.yml # Orchestration with profiles
├── models/                # Shared AI model checkpoints
├── audio/                 # Shared audio files
│   ├── raw/
│   ├── finished/
│   └── stems/
└── README.md
```

### Services Overview

| Service    | GPU | Profile  | Ports | Description |
|------------|-----|----------|-------|-------------|
| `faceswap` | 0   | `stream` | 7860  | Real-time face swapping with DeepFaceLive + InstantID |
| `tracker`  | 0   | `stream` | 7865  | MediaPipe face tracking service |
| `audio`    | 1   | always   | 7861-7863 | TTS, voice conversion, music generation |

### GPU Assignment
- **GPU 0**: Face swapping and tracking (stream mode only)
- **GPU 1**: Audio processing (always available)

## 🚀 Quick Start

### Prerequisites
- Ubuntu 22.04 (or compatible Linux distribution)
- Docker & Docker Compose
- NVIDIA Container Toolkit
- Two NVIDIA GPUs (or modify `docker-compose.yml` for single GPU)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-performance-stack
   ```

2. **Build the containers:**
   ```bash
   cd compose
   docker compose build
   ```

3. **Start services:**

   **Studio Mode (audio only):**
   ```bash
   docker compose up -d audio
   ```

   **Full Stream Rig:**
   ```bash
   docker compose up -d --profile stream
   ```

### First Time Setup

1. **Download AI models** (optional, but recommended):
   ```bash
   # Create models directory structure
   mkdir -p models/{deepfacelive,instantid,musicgen,rvc,bark,xtts,demucs}
   
   # Download sample models (adjust URLs as needed)
   # Example for DeepFaceLive models:
   # wget -O models/deepfacelive/model.pth <model-url>
   ```

2. **Test the services:**
   - Audio Service: http://localhost:7861
   - Face Swap Service: http://localhost:7860 (stream mode only)
   - MediaPipe Tracker: http://localhost:7865 (stream mode only)

## 📝 Usage

### Audio Processing

Access the audio interface at `http://localhost:7861`

**Features:**
- **Music Generation**: Use MusicGen to create music from text prompts
- **Text-to-Speech**: Convert text to speech with multiple TTS models (Bark, XTTS, Tortoise)
- **Voice Conversion**: Change voices using RVC (Retrieval-based Voice Conversion)
- **Audio Separation**: Split songs into stems (vocals, drums, bass, other) using Demucs

**API Endpoints:**
```bash
# Generate music
curl -X POST "http://localhost:7861/generate/music" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "upbeat electronic dance music", "duration": 30}'

# Text-to-speech
curl -X POST "http://localhost:7861/generate/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "default", "model": "bark"}'

# Voice conversion
curl -X POST "http://localhost:7861/convert/voice" \
  -F "file=@input.wav" \
  -F "target_voice=voice1"

# Audio separation
curl -X POST "http://localhost:7861/separate" \
  -F "file=@song.wav" \
  -F "model=htdemucs"
```

### Face Swapping (Stream Mode)

Access the face swap interface at `http://localhost:7860` when running in stream mode.

**Features:**
- Real-time face detection using MediaPipe
- Face swapping with DeepFaceLive
- InstantID integration for identity preservation
- WebSocket support for real-time streaming

**API Endpoints:**
```bash
# Health check
curl http://localhost:7860/health

# List available models
curl http://localhost:7860/models

# WebSocket connection for real-time processing
ws://localhost:7860/ws/faceswap
```

## 🔧 Configuration

### Docker Compose Profiles

**Studio Mode (default):**
- Only runs audio processing services
- Uses GPU 1
- Ideal for music production and voice work

**Stream Mode:**
- Runs all services including face swapping
- Uses both GPUs
- Full streaming setup with real-time capabilities

### Environment Variables

You can customize the setup by modifying environment variables in `compose/docker-compose.yml`:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=1    # GPU assignment
  - HF_HOME=/app/models       # Hugging Face cache directory
  - TORCH_HOME=/app/models    # PyTorch model cache
```

### Volume Mounts

- `../models:/app/models:rw` - Shared model storage
- `../audio:/app/audio:rw` - Shared audio files
- Named volumes for temporary files and caches

## 🛠️ Development

### Adding New Models

1. **Place model files in the appropriate directory:**
   ```bash
   # For face swap models
   cp new_model.pth models/deepfacelive/

   # For audio models
   cp audio_model.pth models/rvc/
   ```

2. **Restart services to load new models:**
   ```bash
   docker compose restart
   ```

### Customizing Services

Each service can be run in different modes:

**Face Swap Service:**
```bash
# Web UI mode (default)
docker compose exec faceswap python faceswap_app.py --mode webui

# API only mode
docker compose exec faceswap python faceswap_app.py --mode api

# Tracker only mode
docker compose exec faceswap python faceswap_app.py --mode tracker
```

**Audio Service:**
```bash
# All interfaces (includes TTS-WebUI and RVC-WebUI)
docker compose exec audio python audio_app.py --mode all

# Main interface only
docker compose exec audio python audio_app.py --mode webui

# API only
docker compose exec audio python audio_app.py --mode api
```

### Building Custom Images

```bash
# Build faceswap service only
docker compose build faceswap

# Build audio service only
docker compose build audio

# Build with no cache
docker compose build --no-cache
```

## 📊 Performance Monitoring

### Resource Usage

Monitor GPU usage:
```bash
# Check GPU utilization
nvidia-smi

# Watch GPU usage continuously
watch -n 1 nvidia-smi
```

Monitor container resources:
```bash
# Container stats
docker stats

# Specific service logs
docker compose logs -f audio
docker compose logs -f faceswap
```

### Health Checks

All services include health check endpoints:
```bash
curl http://localhost:7861/health  # Audio service
curl http://localhost:7860/health  # Face swap service (stream mode)
```

## � Troubleshooting

### Common Issues

**NVIDIA Docker not working:**
```bash
# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Out of GPU memory:**
- Reduce batch sizes in model configurations
- Use smaller models
- Restart services to clear GPU memory

**Model download issues:**
- Check internet connection
- Verify Hugging Face authentication if using gated models
- Manually download models to the `models/` directory

### Logs and Debugging

```bash
# View all service logs
docker compose logs

# Follow logs for specific service
docker compose logs -f audio

# Get into a running container for debugging
docker compose exec audio bash
docker compose exec faceswap bash

# Check container resource usage
docker compose top
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test with both studio and stream profiles
5. Submit a pull request

## � License

[Add your license information here]

## 🙏 Acknowledgments

- **DeepFaceLive** - Real-time face swapping technology
- **InstantID** - Identity preservation for face swapping
- **MediaPipe** - Face detection and tracking
- **MusicGen** - AI music generation by Meta
- **Bark** - Generative audio model
- **XTTS** - Cross-lingual text-to-speech
- **RVC** - Retrieval-based voice conversion
- **Demucs** - Music source separation

---

**Happy streaming and creating! 🎭🎵**
