import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
import tempfile
import traceback
from PIL import Image

class VideoUpscaler:
    """Real-ESRGAN video upscaling integration"""
    
    def __init__(self):
        self.upscaler = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_complete = False
        self.available_models = {
            "RealESRGAN_x4plus": {
                "scale": 4,
                "description": "General purpose 4x upscaling",
                "model_path": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            },
            "RealESRGAN_x2plus": {
                "scale": 2,
                "description": "General purpose 2x upscaling",
                "model_path": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            },
            "RealESRGAN_x4plus_anime_6B": {
                "scale": 4,
                "description": "Anime/artwork 4x upscaling",
                "model_path": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            }
        }
        
        self._setup_realesrgan()
    
    def _setup_realesrgan(self):
        """Setup Real-ESRGAN upscaler"""
        try:
            print("🔧 Setting up Real-ESRGAN upscaler...")
            
            # Try to import Real-ESRGAN
            try:
                from realesrgan import RealESRGANer
                from basicsr.archs.rrdbnet_arch import RRDBNet
                
                # Download and setup default model
                model_name = "RealESRGAN_x4plus"
                model_info = self.available_models[model_name]
                
                model_dir = Path("models/upscaling")
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / f"{model_name}.pth"
                
                # Download model if not exists
                if not model_path.exists():
                    self._download_model(model_info["model_path"], model_path)
                
                # Initialize upscaler
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4
                )
                
                self.upscaler = RealESRGANer(
                    scale=4,
                    model_path=str(model_path),
                    model=model,
                    tile=512,  # Tile size for memory efficiency
                    tile_pad=10,
                    pre_pad=0,
                    half=True if self.device == "cuda" else False,
                    device=self.device
                )
                
                print("✅ Real-ESRGAN initialized successfully")
                self.setup_complete = True
                
            except ImportError as e:
                print(f"⚠️ Real-ESRGAN not available: {e}")
                self._setup_fallback_upscaler()
                
        except Exception as e:
            print(f"❌ Failed to setup Real-ESRGAN: {e}")
            traceback.print_exc()
            self._setup_fallback_upscaler()
    
    def _download_model(self, url: str, output_path: Path):
        """Download Real-ESRGAN model"""
        try:
            print(f"📥 Downloading model to {output_path}...")
            
            import urllib.request
            urllib.request.urlretrieve(url, output_path)
            
            print("✅ Model downloaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            raise
    
    def _setup_fallback_upscaler(self):
        """Setup fallback upscaler using OpenCV"""
        try:
            print("🔧 Setting up fallback upscaler...")
            
            self.upscaler = "opencv_fallback"
            self.setup_complete = True
            
            print("✅ Fallback upscaler setup complete")
            
        except Exception as e:
            print(f"❌ Failed to setup fallback upscaler: {e}")
            self.setup_complete = False
    
    def upscale_video(
        self,
        input_video_path: str,
        output_video_path: str,
        scale_factor: int = 4,
        model_name: str = "RealESRGAN_x4plus",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> str:
        """Upscale video using Real-ESRGAN"""
        
        if not self.setup_complete:
            raise RuntimeError("Upscaler not properly initialized")
        
        try:
            print(f"🎬 Upscaling video: {input_video_path}")
            
            # Open input video
            cap = cv2.VideoCapture(input_video_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate new dimensions
            new_width = original_width * scale_factor
            new_height = original_height * scale_factor
            
            print(f"📐 Original: {original_width}x{original_height}")
            print(f"📐 Upscaled: {new_width}x{new_height}")
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))
            
            if progress_callback:
                progress_callback(0, total_frames, "Starting video upscaling...")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Upscale frame
                if self.upscaler == "opencv_fallback":
                    upscaled_frame = self._upscale_frame_opencv(frame, scale_factor)
                else:
                    upscaled_frame = self._upscale_frame_realesrgan(frame)
                
                # Write frame
                out.write(upscaled_frame)
                
                frame_count += 1
                if progress_callback:
                    progress_callback(
                        frame_count, 
                        total_frames, 
                        f"Upscaling frame {frame_count}/{total_frames}"
                    )
            
            # Cleanup
            cap.release()
            out.release()
            
            if progress_callback:
                progress_callback(total_frames, total_frames, "Upscaling complete!")
            
            print(f"✅ Video upscaled successfully: {output_video_path}")
            return output_video_path
            
        except Exception as e:
            print(f"❌ Video upscaling failed: {e}")
            traceback.print_exc()
            raise
    
    def _upscale_frame_realesrgan(self, frame: np.ndarray) -> np.ndarray:
        """Upscale single frame using Real-ESRGAN"""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Upscale
            upscaled_rgb, _ = self.upscaler.enhance(frame_rgb, outscale=4)
            
            # Convert back to BGR
            upscaled_bgr = cv2.cvtColor(upscaled_rgb, cv2.COLOR_RGB2BGR)
            
            return upscaled_bgr
            
        except Exception as e:
            print(f"⚠️ Real-ESRGAN frame upscaling failed, using fallback: {e}")
            return self._upscale_frame_opencv(frame, 4)
    
    def _upscale_frame_opencv(self, frame: np.ndarray, scale_factor: int) -> np.ndarray:
        """Upscale single frame using OpenCV (fallback)"""
        height, width = frame.shape[:2]
        new_width = width * scale_factor
        new_height = height * scale_factor
        
        # Use INTER_LANCZOS4 for better quality
        upscaled = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        return upscaled
    
    def upscale_image(
        self,
        input_image_path: str,
        output_image_path: str,
        scale_factor: int = 4
    ) -> str:
        """Upscale single image"""
        
        if not self.setup_complete:
            raise RuntimeError("Upscaler not properly initialized")
        
        try:
            # Load image
            image = cv2.imread(input_image_path)
            
            # Upscale
            if self.upscaler == "opencv_fallback":
                upscaled = self._upscale_frame_opencv(image, scale_factor)
            else:
                upscaled = self._upscale_frame_realesrgan(image)
            
            # Save
            cv2.imwrite(output_image_path, upscaled)
            
            print(f"✅ Image upscaled: {output_image_path}")
            return output_image_path
            
        except Exception as e:
            print(f"❌ Image upscaling failed: {e}")
            raise
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available upscaling models"""
        return self.available_models
    
    def switch_model(self, model_name: str):
        """Switch to different upscaling model"""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available")
        
        try:
            model_info = self.available_models[model_name]
            model_dir = Path("models/upscaling")
            model_path = model_dir / f"{model_name}.pth"
            
            # Download if not exists
            if not model_path.exists():
                self._download_model(model_info["model_path"], model_path)
            
            # Reinitialize upscaler with new model
            if self.upscaler != "opencv_fallback":
                from realesrgan import RealESRGANer
                from basicsr.archs.rrdbnet_arch import RRDBNet
                
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=model_info["scale"]
                )
                
                self.upscaler = RealESRGANer(
                    scale=model_info["scale"],
                    model_path=str(model_path),
                    model=model,
                    tile=512,
                    tile_pad=10,
                    pre_pad=0,
                    half=True if self.device == "cuda" else False,
                    device=self.device
                )
            
            print(f"✅ Switched to model: {model_name}")
            
        except Exception as e:
            print(f"❌ Failed to switch model: {e}")
            raise
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get upscaler system information"""
        return {
            "setup_complete": self.setup_complete,
            "device": self.device,
            "upscaler_type": "Real-ESRGAN" if self.upscaler != "opencv_fallback" else "OpenCV Fallback",
            "available_models": list(self.available_models.keys()),
            "cuda_available": torch.cuda.is_available()
        }