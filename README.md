# 🎬 AI Video Generation Studio

A locally hosted AI video generation system powered by **SkyReels v2** that runs efficiently on a single NVIDIA H100 GPU. Generate high-quality videos from text prompts, images, or extend existing videos with full creative freedom and no NSFW filtering.

![AI Video Generation Studio](https://img.shields.io/badge/AI-Video%20Generation-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![CUDA](https://img.shields.io/badge/CUDA-Required-orange) ![License](https://img.shields.io/badge/License-MIT-blue)

## ✨ Features

### 🎭 Generation Modes
- **Text-to-Video**: Create videos from detailed text prompts
- **Image-to-Video**: Transform static images into dynamic videos
- **Video Extension**: Seamlessly extend existing videos with new content

### ⚙️ Advanced Controls
- **Resolution Options**: 540p (960x540) and 720p (1280x720)
- **Frame Control**: 16-128 frames with customizable FPS (8-30)
- **Guidance Scale**: Fine-tune prompt adherence (1.0-20.0)
- **Inference Steps**: Balance quality vs speed (10-50 steps)
- **Seed Control**: Reproducible generation with custom seeds

### 🎨 User Interface
- **Streamlit Web Interface**: Clean, responsive, and user-friendly
- **Real-time Progress**: Live generation progress with time estimates
- **Generation History**: View and manage previous generations
- **System Monitoring**: GPU memory and system resource tracking
- **Inline Video Playback**: Preview results directly in the browser

### 🚀 Performance Features
- **H100 GPU Optimized**: Tailored for NVIDIA H100 performance
- **Memory Efficient**: Smart memory management for large models
- **Background Processing**: Non-blocking video generation
- **Automatic Cleanup**: Temporary file and memory management

## 🛠️ System Requirements

### Hardware Requirements
- **GPU**: NVIDIA H100 (recommended) or any CUDA-compatible GPU with 8GB+ VRAM
- **RAM**: 32GB+ recommended (16GB minimum)
- **Storage**: 100GB+ free space for models and outputs
- **CPU**: Modern multi-core processor

### Software Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.8 or higher
- **CUDA**: 11.8+ or 12.1+
- **Git**: For repository cloning
- **FFmpeg**: For video processing

## 📦 Installation

### Quick Setup

1. **Clone the repository**:
```bash
git clone https://github.com/tjmance/mysky.git
cd ai-video-generation-studio
```

2. **Run the automated setup**:
```bash
python3 setup.py
```

3. **Launch the application**:
```bash
./launch.sh
```

4. **Open in browser**:
Navigate to `http://localhost:8501`

### Manual Installation

1. **Install system dependencies**:
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y ffmpeg git python3-pip

# CentOS/RHEL
sudo yum install -y ffmpeg git python3-pip

# Arch Linux
sudo pacman -S ffmpeg git python
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install PyTorch with CUDA**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. **Setup directories**:
```bash
mkdir -p outputs models temp uploads
```

5. **Download SkyReels (optional)**:
```bash
git clone https://huggingface.co/Skywork/SkyReels-V2-I2V-14B-720P
```

## 🎮 Usage

### Starting the Application

**Option 1: Using the launch script**
```bash
./launch.sh
```

**Option 2: Direct Streamlit launch**
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

**Option 3: Custom configuration**
```bash
streamlit run app.py --server.port 8080 --server.maxUploadSize 500
```

### Generation Workflow

1. **Select Generation Mode**:
   - Choose Text-to-Video, Image-to-Video, or Video Extension

2. **Configure Settings**:
   - Set resolution (540p or 720p)
   - Choose number of frames (16-128)
   - Set FPS (8-30)
   - Adjust advanced parameters in sidebar

3. **Input Content**:
   - Enter detailed text prompt
   - Add negative prompt (optional)
   - Upload image/video if required

4. **Generate Video**:
   - Click "Generate Video" button
   - Monitor progress in real-time
   - View result when complete

### Example Prompts

**Text-to-Video Examples**:
```
A majestic eagle soaring over snow-capped mountains at golden hour, cinematic lighting, 4K quality

A bustling cyberpunk street scene with neon lights reflecting on wet pavement, rain falling, futuristic cars

Time-lapse of a blooming flower garden transitioning from dawn to dusk, vibrant colors, nature documentary style
```

**Negative Prompt Examples**:
```
blurry, low quality, distorted, pixelated, watermark, text, bad anatomy, deformed

static, boring, monochrome, low resolution, artifacts, noise, compression artifacts
```

## 📁 Project Structure

```
ai-video-generation-studio/
├── app.py                 # Main Streamlit application
├── setup.py              # Automated setup script
├── launch.sh             # Launch script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── backend/             # Backend modules
│   ├── __init__.py
│   ├── video_generator.py    # Main generation orchestrator
│   ├── skyreels_integration.py  # SkyReels v2 integration
│   └── utils.py         # Utility functions
├── outputs/             # Generated videos
│   └── metadata/        # Generation metadata
├── models/              # AI models
│   └── SkyReels/        # SkyReels v2 repository
├── temp/                # Temporary files
└── uploads/             # User uploaded files
```

## ⚙️ Configuration

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Memory optimization
```

### Custom Model Paths
```python
# In backend/skyreels_integration.py
MODEL_PATH = "path/to/custom/model"
```

### Streamlit Configuration
Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
address = "0.0.0.0"
maxUploadSize = 500

[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
```

## 🔧 Advanced Features

### Planned Enhancements

1. **Real-ESRGAN Integration**:
   - Upscale generated videos to higher resolutions
   - Enhance video quality and detail

2. **Frame Interpolation**:
   - RIFE/Flowframes support for smooth motion
   - Increase FPS for smoother playback

3. **ControlNet Support**:
   - Pose and depth control for precise direction
   - Enhanced creative control over generation

4. **Multi-GPU Scaling**:
   - Torchrun support for distributed inference
   - Faster generation on multiple GPUs

5. **Docker Containerization**:
   - Easy deployment with Docker
   - Consistent environment across systems

### API Integration

The system can be extended with a REST API:

```python
# Future API endpoint structure
POST /api/generate
{
    "prompt": "Your video description",
    "mode": "Text-to-Video",
    "resolution": "720p",
    "frames": 64,
    "fps": 24
}
```

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**:
```bash
# Reduce batch size or resolution
# Enable sequential CPU offload in config
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

**SkyReels Import Error**:
```bash
# Reinstall SkyReels repository
rm -rf models/SkyReels
git clone https://github.com/SkyworkAI/SkyReels.git models/SkyReels
```

**Streamlit Port Conflict**:
```bash
# Use different port
streamlit run app.py --server.port 8502
```

**FFmpeg Not Found**:
```bash
# Install FFmpeg
sudo apt install ffmpeg  # Ubuntu/Debian
sudo yum install ffmpeg  # CentOS/RHEL
```

### Performance Optimization

1. **GPU Memory Optimization**:
   - Use FP16 precision for inference
   - Enable memory-efficient attention
   - Clear CUDA cache between generations

2. **System Optimization**:
   - Close unnecessary applications
   - Use fast SSD storage for models
   - Ensure adequate cooling for sustained performance

## 📊 System Monitoring

The application includes built-in monitoring:

- **GPU Memory Usage**: Real-time VRAM tracking
- **System Resources**: CPU, RAM, and disk usage
- **Generation Statistics**: Success rates and timing
- **Queue Management**: Active and pending jobs

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
# Clone for development
git clone <repository-url>
cd ai-video-generation-studio

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SkyworkAI** for the SkyReels v2 model
- **Streamlit** for the web framework
- **PyTorch** for the deep learning foundation
- **NVIDIA** for CUDA and GPU acceleration

## 📞 Support

- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Community discussions on GitHub Discussions
- **Documentation**: Check the Wiki for detailed guides

## 🚀 Quick Start Examples

### Text-to-Video Generation
```python
# Example using the backend directly
from backend.video_generator import VideoGenerator

generator = VideoGenerator()
result = generator.generate_async(
    job_id="test_001",
    params={
        "prompt": "A serene mountain lake at sunset",
        "mode": "Text-to-Video",
        "resolution": "720p (1280x720)",
        "num_frames": 64,
        "fps": 24
    }
)
```

### Image-to-Video Generation
```python
# Generate video from uploaded image
result = generator.generate_async(
    job_id="test_002",
    params={
        "prompt": "The landscape comes alive with flowing water",
        "mode": "Image-to-Video",
        "resolution": "720p (1280x720)",
        "num_frames": 48,
        "fps": 24
    },
    input_file=Path("uploads/landscape.jpg")
)
```

---

**🎬 Start creating amazing videos with AI today!**

For more information, visit our [documentation](docs/) or check out the [examples](examples/) directory.
