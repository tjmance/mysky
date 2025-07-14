# 🎉 AI Performance Stack Setup Complete!

The **AI Performance Stack** monorepo has been successfully created with all components configured and ready for deployment.

## 📁 Repository Structure

```
ai-performance-stack/
├── 📄 README.md                           # Comprehensive documentation
├── 🔧 setup.sh                           # Automated setup script  
├── 📄 .gitignore                         # Git ignore rules
├── 📄 docker-compose.override.yml.example # Local customization template
├── 🐳 docker/                            # Docker configurations
│   ├── faceswap/                         # Face swapping service
│   │   ├── Dockerfile                    # GPU 0 optimized
│   │   ├── faceswap_app.py              # Main application
│   │   └── requirements.txt             # Additional Python packages
│   └── audio/                           # Audio processing service  
│       ├── Dockerfile                    # GPU 1 optimized
│       ├── audio_app.py                 # Main application
│       └── requirements.txt             # Additional Python packages
├── 🎭 compose/                           # Docker Compose orchestration
│   └── docker-compose.yml              # Service definitions & profiles
├── 🧠 models/                            # AI model storage
│   └── .gitkeep                         # Preserve directory structure
└── 🎵 audio/                            # Audio file storage
    ├── raw/                             # Input audio files
    ├── finished/                        # Processed outputs
    ├── stems/                           # Separated audio stems
    └── .gitkeep                         # Preserve directory structure
```

## ✅ What's Been Created

### 🐳 Docker Services

1. **FaceSwap Service (`faceswap`)**
   - **Base**: NVIDIA CUDA 11.8 Ubuntu 22.04
   - **GPU**: 0 (stream profile only)
   - **Port**: 7860
   - **Features**: DeepFaceLive, InstantID, MediaPipe
   - **Modes**: WebUI, API, Tracker

2. **MediaPipe Tracker (`tracker`)**
   - **GPU**: 0 (stream profile only)  
   - **Port**: 7865
   - **Purpose**: Dedicated face tracking service

3. **Audio Service (`audio`)**
   - **Base**: NVIDIA CUDA 11.8 Ubuntu 22.04
   - **GPU**: 1 (always available)
   - **Ports**: 7861, 7862, 7863
   - **Features**: MusicGen, TTS (Bark/XTTS), RVC, Demucs
   - **Modes**: WebUI, API, All (includes sub-UIs)

### 🔧 Configuration Features

- **Profile-based Deployment**: Studio vs Stream modes
- **GPU Assignment**: Automatic GPU 0/1 allocation
- **Health Checks**: Built-in service monitoring
- **Volume Mounts**: Shared model and audio storage
- **Override Support**: Local customization without Git conflicts

### 📋 Service Profiles

| Profile | Services | Use Case | Command |
|---------|----------|----------|---------|
| **Studio** | `audio` only | Music production, voice work | `docker compose up -d audio` |
| **Stream** | All services | Live streaming with face/voice | `docker compose up -d --profile stream` |

## 🚀 Next Steps

### 1. **Run Setup Script**
```bash
# Make executable and run automated setup
chmod +x setup.sh
./setup.sh
```

The setup script will:
- ✅ Check OS compatibility (Ubuntu 22.04)
- ✅ Verify hardware requirements (NVIDIA GPUs)
- ✅ Install Docker if needed
- ✅ Install NVIDIA Container Toolkit
- ✅ Build Docker images
- ✅ Test the installation

### 2. **Download AI Models** (Optional but Recommended)

```bash
# Example model downloads (adjust URLs as needed)
mkdir -p models/{deepfacelive,instantid,musicgen,rvc,bark,xtts,demucs}

# DeepFaceLive models
# wget -O models/deepfacelive/model.pth <model-url>

# InstantID models  
# wget -O models/instantid/model.safetensors <model-url>

# RVC models
# wget -O models/rvc/voice_model.pth <model-url>
```

### 3. **Launch Services**

**For Studio Work (Audio Only):**
```bash
cd compose
docker compose up -d audio
```

**For Full Streaming Setup:**
```bash
cd compose  
docker compose up -d --profile stream
```

### 4. **Access Web Interfaces**

| Service | URL | Description |
|---------|-----|-------------|
| **Audio Processing** | http://localhost:7861 | Main audio interface |
| **TTS WebUI** | http://localhost:7862 | Text-to-speech interface |
| **RVC WebUI** | http://localhost:7863 | Voice conversion interface |
| **Face Swapping** | http://localhost:7860 | Face swap interface (stream mode) |
| **Face Tracking** | http://localhost:7865 | MediaPipe tracker (stream mode) |

## 🔍 Verification Commands

```bash
# Check service health
curl http://localhost:7861/health  # Audio service
curl http://localhost:7860/health  # Face swap (if running)

# Monitor GPU usage
nvidia-smi

# View logs
docker compose logs -f audio
docker compose logs -f faceswap  # If running

# Check running containers
docker compose ps
```

## 🛠️ Customization

### Single GPU Setup
If you only have one GPU, copy and modify the override file:
```bash
cp docker-compose.override.yml.example docker-compose.override.yml
# Edit the file to use CUDA_VISIBLE_DEVICES=0 for all services
```

### Custom Ports
Modify `docker-compose.override.yml` to change port mappings:
```yaml
services:
  audio:
    ports:
      - "8861:7861"  # Custom port mapping
```

## 📊 Performance Notes

- **Audio Service**: Uses GPU 1, requires ~8GB VRAM for large models
- **Face Swap**: Uses GPU 0, requires ~6GB VRAM  
- **Combined Load**: Recommend 16GB+ total VRAM for full stack
- **Memory**: 16GB+ RAM recommended, 32GB+ ideal

## 🤝 Getting Help

1. **Check logs**: `docker compose logs <service-name>`
2. **Restart services**: `docker compose restart`
3. **Rebuild images**: `docker compose build --no-cache`
4. **Health checks**: Use `/health` endpoints
5. **Documentation**: See `README.md` for detailed usage

## 🎯 What You Can Do Now

### Audio Generation
- 🎵 Generate music from text prompts
- 🗣️ Convert text to speech with multiple voices
- 🎤 Change voices in real-time or offline
- 🎛️ Separate songs into individual stems

### Face Processing (Stream Mode)
- 👤 Real-time face detection and tracking  
- 🎭 Swap faces between images or video streams
- 📱 WebSocket support for live applications
- 🎥 Identity preservation with InstantID

---

**🎉 Your AI Performance Stack is ready! Start creating amazing content with real-time AI tools.**

For detailed usage instructions, API documentation, and troubleshooting, see the main [README.md](README.md).