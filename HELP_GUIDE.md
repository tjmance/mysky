# 🆘 AI Video Generation Studio - Help Guide

## 🚀 Quick Start

### 1. First Time Setup
```bash
# Install dependencies
pip install -r requirements.txt

# For advanced features
pip install -r requirements-advanced.txt

# Make launch script executable
chmod +x launch.sh
```

### 2. Launch the Application
```bash
# Simple launch
python app.py

# Or use the launch script
./launch.sh

# For distributed setup
python launch_distributed.py
```

### 3. Access the Web Interface
- Open your browser and go to: `http://localhost:8501`
- The Streamlit interface should load automatically

## 🛠️ Common Issues & Solutions

### GPU Issues
**Problem**: "CUDA out of memory"
- **Solution**: Reduce resolution (try 540p instead of 720p)
- **Solution**: Decrease number of frames
- **Solution**: Lower inference steps

**Problem**: "No CUDA device found"
- **Solution**: Check GPU drivers: `nvidia-smi`
- **Solution**: Verify CUDA installation: `nvcc --version`

### Installation Issues
**Problem**: Package conflicts
- **Solution**: Use virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux
pip install -r requirements.txt
```

**Problem**: Missing dependencies
- **Solution**: Install system dependencies:
```bash
sudo apt update
sudo apt install ffmpeg git python3-dev
```

### Performance Issues
**Problem**: Slow generation
- **Solution**: Reduce inference steps (try 20-30)
- **Solution**: Use smaller resolution
- **Solution**: Check GPU memory usage in interface

## 🎬 Usage Guide

### Text-to-Video Generation
1. Select "Text-to-Video" mode
2. Enter detailed prompt (be specific!)
3. Choose resolution and frame count
4. Adjust guidance scale (7.5 is usually good)
5. Click "Generate Video"

### Image-to-Video Generation
1. Select "Image-to-Video" mode
2. Upload your image (JPG/PNG)
3. Add motion description in prompt
4. Configure settings
5. Generate

### Video Extension
1. Select "Video Extension" mode
2. Upload existing video
3. Describe what should happen next
4. Set frame count for extension
5. Generate

## 🔧 Advanced Configuration

### Model Settings
- **Guidance Scale**: Higher = more prompt adherence (7.5-15.0 recommended)
- **Inference Steps**: More steps = better quality but slower (20-50)
- **Seed**: Use same seed for reproducible results

### Performance Tuning
- Monitor GPU memory in the interface
- Use automatic cleanup to free memory
- Batch smaller generations if memory is limited

## 📁 File Locations

### Generated Videos
- Default output: `./outputs/`
- Videos saved with timestamp and settings

### Models
- Models downloaded to: `./models/`
- First run will download ~15GB of models

### Logs
- Application logs: Check terminal output
- Error logs: Saved to `./logs/` (if configured)

## 🐛 Debugging

### Enable Debug Mode
```bash
# Run with debug logging
python app.py --debug

# Or set environment variable
export DEBUG=1
python app.py
```

### Check System Status
```bash
# Test the system
python test_system.py

# Check GPU memory
nvidia-smi

# Check dependencies
pip list | grep torch
```

### Common Error Solutions

**"Model not found"**
- Wait for initial model download (first run only)
- Check internet connection
- Verify disk space (need 100GB+)

**"Generation failed"**
- Check prompt length (not too long)
- Verify input image format
- Check available GPU memory

**"Interface won't load"**
- Check if port 8501 is available
- Try different port: `streamlit run app.py --server.port 8502`
- Check firewall settings

## 🆘 Getting More Help

### System Information
Run this to get system info for troubleshooting:
```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'GPU Count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

### Log Analysis
- Check terminal output for errors
- Look for CUDA/memory warnings
- Note any model loading issues

### Performance Benchmarks
- 540p, 16 frames: ~30-60 seconds on H100
- 720p, 32 frames: ~2-5 minutes on H100
- Times vary based on prompt complexity

## 📚 Documentation Files

- `README.md` - Full installation and feature guide
- `SYSTEM_OVERVIEW.md` - Technical architecture details
- `ADVANCED_FEATURES.md` - Advanced usage and customization
- `IMPLEMENTATION_COMPLETE.md` - Development and implementation notes

---

**Need specific help?** Check the relevant documentation file above or examine the error messages in your terminal for more detailed troubleshooting information.