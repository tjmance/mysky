import os
import sys
import subprocess
import torch
import psutil
import platform
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import time

def setup_environment():
    """Setup the environment for video generation"""
    try:
        print("🔧 Setting up environment...")
        
        # Set environment variables for optimal GPU performance
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        # Set memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Create necessary directories
        directories = ["outputs", "models", "temp", "uploads"]
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        
        print("✅ Environment setup complete")
        
    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        raise

def validate_cuda() -> bool:
    """Validate CUDA availability and GPU compatibility"""
    try:
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            print("❌ No CUDA devices found")
            return False
        
        # Get GPU information
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        print(f"✅ GPU: {gpu_name}")
        print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
        
        # Check if it's an H100 or similar high-end GPU
        is_h100 = "H100" in gpu_name
        is_high_memory = gpu_memory >= 40  # At least 40GB for video generation
        
        if is_h100:
            print("✅ H100 GPU detected - optimal for video generation")
        elif is_high_memory:
            print("✅ High-memory GPU detected - suitable for video generation")
        else:
            print(f"⚠️ GPU may have limited memory for video generation ({gpu_memory:.1f} GB)")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA validation failed: {e}")
        return False

def get_device_info() -> Dict[str, Any]:
    """Get detailed device information"""
    info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_cores": psutil.cpu_count(),
        "ram_gb": psutil.virtual_memory().total / 1024**3,
    }
    
    if torch.cuda.is_available():
        info.update({
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "cuda_version": torch.version.cuda,
        })
    
    return info

def get_available_models() -> List[str]:
    """Get list of available video generation models"""
    models = ["SkyReels v2"]
    
    # Check for additional models in the models directory
    models_dir = Path("models")
    if models_dir.exists():
        for model_path in models_dir.iterdir():
            if model_path.is_dir() and model_path.name not in ["SkyReels"]:
                models.append(model_path.name)
    
    return models

def validate_inputs(params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate generation parameters"""
    try:
        # Check required parameters
        required_params = ["prompt", "mode", "resolution", "num_frames", "fps"]
        for param in required_params:
            if param not in params or params[param] is None:
                return False, f"Missing required parameter: {param}"
        
        # Validate prompt
        if not params["prompt"].strip():
            return False, "Prompt cannot be empty"
        
        if len(params["prompt"]) > 1000:
            return False, "Prompt too long (max 1000 characters)"
        
        # Validate mode
        valid_modes = ["Text-to-Video", "Image-to-Video", "Video Extension"]
        if params["mode"] not in valid_modes:
            return False, f"Invalid mode. Must be one of: {valid_modes}"
        
        # Validate resolution
        valid_resolutions = ["540p (960x540)", "720p (1280x720)"]
        if params["resolution"] not in valid_resolutions:
            return False, f"Invalid resolution. Must be one of: {valid_resolutions}"
        
        # Validate numeric parameters
        if not isinstance(params["num_frames"], int) or params["num_frames"] < 8 or params["num_frames"] > 256:
            return False, "Number of frames must be between 8 and 256"
        
        if not isinstance(params["fps"], int) or params["fps"] < 1 or params["fps"] > 60:
            return False, "FPS must be between 1 and 60"
        
        if "guidance_scale" in params:
            if not isinstance(params["guidance_scale"], (int, float)) or params["guidance_scale"] < 1.0 or params["guidance_scale"] > 20.0:
                return False, "Guidance scale must be between 1.0 and 20.0"
        
        if "num_inference_steps" in params:
            if not isinstance(params["num_inference_steps"], int) or params["num_inference_steps"] < 5 or params["num_inference_steps"] > 100:
                return False, "Number of inference steps must be between 5 and 100"
        
        return True, None
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {remaining_minutes}m {remaining_seconds:.1f}s"

def get_system_status() -> Dict[str, Any]:
    """Get current system status"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / 1024**3
        memory_total_gb = memory.total / 1024**3
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used_gb = disk.used / 1024**3
        disk_total_gb = disk.total / 1024**3
        
        status = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_used_gb": memory_used_gb,
            "memory_total_gb": memory_total_gb,
            "disk_percent": disk_percent,
            "disk_used_gb": disk_used_gb,
            "disk_total_gb": disk_total_gb,
            "timestamp": time.time()
        }
        
        # GPU status if available
        if torch.cuda.is_available():
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                status.update({
                    "gpu_memory_allocated_gb": gpu_memory_allocated,
                    "gpu_memory_reserved_gb": gpu_memory_reserved,
                    "gpu_memory_total_gb": gpu_memory_total,
                    "gpu_memory_percent": (gpu_memory_reserved / gpu_memory_total) * 100
                })
            except Exception:
                pass
        
        return status
        
    except Exception as e:
        return {"error": str(e), "timestamp": time.time()}

def cleanup_temp_files(max_age_hours: int = 24):
    """Clean up temporary files older than specified age"""
    try:
        temp_dir = Path("temp")
        if not temp_dir.exists():
            return 0
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0
        
        for file_path in temp_dir.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")
        
        return cleaned_count
        
    except Exception as e:
        print(f"Cleanup failed: {e}")
        return 0

def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available"""
    dependencies = {
        "torch": False,
        "opencv": False,
        "PIL": False,
        "numpy": False,
        "streamlit": False,
        "ffmpeg": False
    }
    
    try:
        import torch
        dependencies["torch"] = True
    except ImportError:
        pass
    
    try:
        import cv2
        dependencies["opencv"] = True
    except ImportError:
        pass
    
    try:
        from PIL import Image
        dependencies["PIL"] = True
    except ImportError:
        pass
    
    try:
        import numpy
        dependencies["numpy"] = True
    except ImportError:
        pass
    
    try:
        import streamlit
        dependencies["streamlit"] = True
    except ImportError:
        pass
    
    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        dependencies["ffmpeg"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return dependencies

def install_missing_dependencies():
    """Install missing dependencies"""
    try:
        print("📦 Installing missing dependencies...")
        
        # Install pip packages
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def save_generation_metadata(job_id: str, params: Dict[str, Any], output_path: str):
    """Save metadata for a completed generation"""
    try:
        metadata_dir = Path("outputs") / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        metadata = {
            "job_id": job_id,
            "timestamp": time.time(),
            "params": params,
            "output_path": output_path,
            "system_info": get_device_info()
        }
        
        metadata_file = metadata_dir / f"{job_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
    except Exception as e:
        print(f"Failed to save metadata: {e}")

def load_generation_metadata(job_id: str) -> Optional[Dict[str, Any]]:
    """Load metadata for a generation"""
    try:
        metadata_file = Path("outputs") / "metadata" / f"{job_id}.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Failed to load metadata: {e}")
    
    return None

def get_output_files() -> List[Dict[str, Any]]:
    """Get list of output files with metadata"""
    outputs = []
    output_dir = Path("outputs")
    
    if not output_dir.exists():
        return outputs
    
    for file_path in output_dir.glob("*.mp4"):
        try:
            stat = file_path.stat()
            file_info = {
                "filename": file_path.name,
                "path": str(file_path),
                "size_mb": stat.st_size / 1024**2,
                "created": stat.st_ctime,
                "modified": stat.st_mtime
            }
            
            # Try to load associated metadata
            job_id = file_path.stem.split('_')[-1] if '_' in file_path.stem else None
            if job_id:
                metadata = load_generation_metadata(job_id)
                if metadata:
                    file_info["metadata"] = metadata
            
            outputs.append(file_info)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Sort by creation time (newest first)
    outputs.sort(key=lambda x: x.get("created", 0), reverse=True)
    
    return outputs