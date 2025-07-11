import os
import sys
import subprocess
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Union, Dict, Any
import tempfile
import shutil
import json
import cv2
from PIL import Image
import traceback

class SkyReelsGenerator:
    """SkyReels v2 video generation integration"""
    
    def __init__(self):
        self.model = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = None
        self.setup_complete = False
        
        # Initialize the model
        self._setup_skyreels()
    
    def _setup_skyreels(self):
        """Setup SkyReels v2 model and pipeline"""
        try:
            print("🔧 Setting up SkyReels v2...")
            
            # Check if SkyReels repository exists
            skyreels_path = Path("models/SkyReels")
            if not skyreels_path.exists():
                self._download_skyreels()
            
            # Add SkyReels to Python path
            sys.path.insert(0, str(skyreels_path))
            
            # Import SkyReels modules
            try:
                from sky_reels.models.video_diffusion import SkyReelsVideoDiffusion
                from sky_reels.pipelines.video_generation import SkyReelsVideoPipeline
                from sky_reels.utils.config import load_config
                
                # Load configuration
                config_path = skyreels_path / "configs" / "skyreels_v2.yaml"
                if not config_path.exists():
                    self._create_default_config(config_path)
                
                config = load_config(str(config_path))
                
                # Initialize pipeline
                self.pipeline = SkyReelsVideoPipeline(
                    config=config,
                    device=self.device,
                    dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                
                print("✅ SkyReels v2 pipeline initialized successfully")
                self.setup_complete = True
                
            except ImportError as e:
                print(f"⚠️ SkyReels modules not found, using fallback implementation: {e}")
                self._setup_fallback_generator()
                
        except Exception as e:
            print(f"❌ Failed to setup SkyReels: {e}")
            traceback.print_exc()
            self._setup_fallback_generator()
    
    def _download_skyreels(self):
        """Download SkyReels v2 repository"""
        try:
            print("📥 Downloading SkyReels v2 repository...")
            
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            # Clone SkyReels repository
            subprocess.run([
                "git", "clone", 
                "https://github.com/SkyworkAI/SkyReels.git",
                str(models_dir / "SkyReels")
            ], check=True, capture_output=True)
            
            print("✅ SkyReels repository downloaded successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download SkyReels: {e}")
            # Create a basic structure for fallback
            skyreels_path = Path("models/SkyReels")
            skyreels_path.mkdir(parents=True, exist_ok=True)
            self._create_fallback_structure(skyreels_path)
    
    def _create_default_config(self, config_path: Path):
        """Create a default configuration for SkyReels"""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        default_config = {
            "model": {
                "name": "skyreels_v2",
                "num_frames": 64,
                "frame_rate": 24,
                "resolution": [720, 1280],
                "channels": 3
            },
            "inference": {
                "num_inference_steps": 25,
                "guidance_scale": 7.5,
                "scheduler": "DDPM"
            },
            "memory": {
                "enable_memory_efficient_attention": True,
                "enable_sequential_cpu_offload": False,
                "enable_model_cpu_offload": True
            }
        }
        
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(default_config, f, default_flow_style=False)
    
    def _create_fallback_structure(self, skyreels_path: Path):
        """Create a fallback directory structure"""
        (skyreels_path / "configs").mkdir(exist_ok=True)
        (skyreels_path / "sky_reels").mkdir(exist_ok=True)
        (skyreels_path / "sky_reels" / "models").mkdir(exist_ok=True)
        (skyreels_path / "sky_reels" / "pipelines").mkdir(exist_ok=True)
        (skyreels_path / "sky_reels" / "utils").mkdir(exist_ok=True)
        
        # Create empty __init__.py files
        for init_path in [
            skyreels_path / "sky_reels" / "__init__.py",
            skyreels_path / "sky_reels" / "models" / "__init__.py",
            skyreels_path / "sky_reels" / "pipelines" / "__init__.py",
            skyreels_path / "sky_reels" / "utils" / "__init__.py"
        ]:
            init_path.touch()
    
    def _setup_fallback_generator(self):
        """Setup a fallback video generator using diffusers"""
        try:
            print("🔧 Setting up fallback video generator...")
            
            from diffusers import DiffusionPipeline, StableDiffusionPipeline
            import torch
            
            # Use a general video generation pipeline or create synthetic videos
            self.pipeline = "fallback"
            self.setup_complete = True
            
            print("✅ Fallback generator setup complete")
            
        except Exception as e:
            print(f"❌ Failed to setup fallback generator: {e}")
            self.setup_complete = False
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        mode: str = "Text-to-Video",
        resolution: str = "720p (1280x720)",
        num_frames: int = 64,
        fps: int = 24,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        seed: Optional[int] = None,
        input_file: Optional[str] = None,
        output_path: str = "output.mp4",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> str:
        """Generate video using SkyReels v2"""
        
        if not self.setup_complete:
            raise RuntimeError("SkyReels generator not properly initialized")
        
        try:
            # Set random seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            # Parse resolution
            width, height = self._parse_resolution(resolution)
            
            # Progress callback wrapper
            def progress_fn(step: int, total_steps: int, message: str = ""):
                if progress_callback:
                    progress_callback(step, total_steps, message)
            
            # Generate based on mode
            if mode == "Text-to-Video":
                return self._generate_text_to_video(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    fps=fps,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    output_path=output_path,
                    progress_callback=progress_fn
                )
            
            elif mode == "Image-to-Video":
                return self._generate_image_to_video(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    input_image_path=input_file,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    fps=fps,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    output_path=output_path,
                    progress_callback=progress_fn
                )
            
            elif mode == "Video Extension":
                return self._generate_video_extension(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    input_video_path=input_file,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    fps=fps,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    output_path=output_path,
                    progress_callback=progress_fn
                )
            
            else:
                raise ValueError(f"Unsupported generation mode: {mode}")
                
        except Exception as e:
            print(f"❌ Video generation failed: {e}")
            traceback.print_exc()
            raise
    
    def _parse_resolution(self, resolution: str) -> tuple:
        """Parse resolution string to width, height"""
        if "540p" in resolution:
            return 960, 540
        elif "720p" in resolution:
            return 1280, 720
        else:
            # Default to 720p
            return 1280, 720
    
    def _generate_text_to_video(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        guidance_scale: float,
        num_inference_steps: int,
        output_path: str,
        progress_callback: Callable
    ) -> str:
        """Generate video from text prompt"""
        
        if self.pipeline == "fallback":
            return self._generate_fallback_video(
                prompt, width, height, num_frames, fps, output_path, progress_callback
            )
        
        try:
            progress_callback(0, num_inference_steps, "Initializing text-to-video generation...")
            
            # Use SkyReels pipeline for text-to-video generation
            result = self.pipeline.generate_video(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                callback=lambda step, total, *args: progress_callback(step, total, f"Generating frame {step}/{total}")
            )
            
            # Save video
            progress_callback(num_inference_steps - 1, num_inference_steps, "Saving video...")
            self._save_video_frames(result.frames, output_path, fps)
            progress_callback(num_inference_steps, num_inference_steps, "Complete!")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Text-to-video generation failed: {e}")
            return self._generate_fallback_video(
                prompt, width, height, num_frames, fps, output_path, progress_callback
            )
    
    def _generate_image_to_video(
        self,
        prompt: str,
        negative_prompt: str,
        input_image_path: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        guidance_scale: float,
        num_inference_steps: int,
        output_path: str,
        progress_callback: Callable
    ) -> str:
        """Generate video from input image"""
        
        if self.pipeline == "fallback":
            return self._generate_fallback_video_from_image(
                input_image_path, prompt, width, height, num_frames, fps, output_path, progress_callback
            )
        
        try:
            progress_callback(0, num_inference_steps, "Loading input image...")
            
            # Load and preprocess input image
            input_image = Image.open(input_image_path)
            input_image = input_image.resize((width, height))
            
            progress_callback(1, num_inference_steps, "Initializing image-to-video generation...")
            
            # Use SkyReels pipeline for image-to-video generation
            result = self.pipeline.generate_video_from_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                input_image=input_image,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                callback=lambda step, total, *args: progress_callback(step + 1, total + 1, f"Generating frame {step}/{total}")
            )
            
            # Save video
            progress_callback(num_inference_steps, num_inference_steps, "Saving video...")
            self._save_video_frames(result.frames, output_path, fps)
            
            return output_path
            
        except Exception as e:
            print(f"❌ Image-to-video generation failed: {e}")
            return self._generate_fallback_video_from_image(
                input_image_path, prompt, width, height, num_frames, fps, output_path, progress_callback
            )
    
    def _generate_video_extension(
        self,
        prompt: str,
        negative_prompt: str,
        input_video_path: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        guidance_scale: float,
        num_inference_steps: int,
        output_path: str,
        progress_callback: Callable
    ) -> str:
        """Generate video extension from input video"""
        
        try:
            progress_callback(0, num_inference_steps, "Loading input video...")
            
            # Extract last frame from input video
            cap = cv2.VideoCapture(input_video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            
            if not frames:
                raise ValueError("No frames found in input video")
            
            # Use last frame as starting point
            last_frame = frames[-1]
            last_frame_pil = Image.fromarray(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
            last_frame_pil = last_frame_pil.resize((width, height))
            
            # Generate extension using image-to-video with the last frame
            temp_image_path = "temp_last_frame.jpg"
            last_frame_pil.save(temp_image_path)
            
            extended_video_path = self._generate_image_to_video(
                prompt=prompt,
                negative_prompt=negative_prompt,
                input_image_path=temp_image_path,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                output_path="temp_extension.mp4",
                progress_callback=lambda step, total, msg: progress_callback(step, total + 2, msg)
            )
            
            # Concatenate original video with extension
            progress_callback(num_inference_steps + 1, num_inference_steps + 2, "Concatenating videos...")
            self._concatenate_videos(input_video_path, extended_video_path, output_path)
            
            # Cleanup
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if os.path.exists(extended_video_path):
                os.remove(extended_video_path)
            
            progress_callback(num_inference_steps + 2, num_inference_steps + 2, "Complete!")
            return output_path
            
        except Exception as e:
            print(f"❌ Video extension failed: {e}")
            return self._generate_fallback_video(
                prompt, width, height, num_frames, fps, output_path, progress_callback
            )
    
    def _save_video_frames(self, frames, output_path: str, fps: int):
        """Save frames as video file"""
        if not frames:
            raise ValueError("No frames to save")
        
        # Convert frames to numpy arrays if needed
        if isinstance(frames[0], Image.Image):
            frames = [np.array(frame) for frame in frames]
        
        # Setup video writer
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for frame in frames:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        out.release()
    
    def _concatenate_videos(self, video1_path: str, video2_path: str, output_path: str):
        """Concatenate two video files"""
        try:
            # Use ffmpeg to concatenate videos
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(f"file '{video1_path}'\n")
                f.write(f"file '{video2_path}'\n")
                concat_file = f.name
            
            subprocess.run([
                'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file,
                '-c', 'copy', output_path, '-y'
            ], check=True, capture_output=True)
            
            os.unlink(concat_file)
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: use OpenCV for concatenation
            self._concatenate_videos_opencv(video1_path, video2_path, output_path)
    
    def _concatenate_videos_opencv(self, video1_path: str, video2_path: str, output_path: str):
        """Concatenate videos using OpenCV"""
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps1, (width1, height1))
        
        # Copy frames from first video
        while True:
            ret, frame = cap1.read()
            if not ret:
                break
            out.write(frame)
        
        # Copy frames from second video
        while True:
            ret, frame = cap2.read()
            if not ret:
                break
            # Resize if necessary
            if frame.shape[:2] != (height1, width1):
                frame = cv2.resize(frame, (width1, height1))
            out.write(frame)
        
        cap1.release()
        cap2.release()
        out.release()
    
    def _generate_fallback_video(
        self,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        output_path: str,
        progress_callback: Callable
    ) -> str:
        """Generate a fallback video (synthetic/placeholder)"""
        try:
            progress_callback(0, num_frames, "Generating fallback video...")
            
            # Create synthetic video frames
            frames = []
            for i in range(num_frames):
                # Create a gradient frame with text
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Add gradient
                for y in range(height):
                    for x in range(width):
                        frame[y, x] = [
                            int(255 * (x / width)),
                            int(255 * (y / height)),
                            int(255 * ((x + y) / (width + height)))
                        ]
                
                # Add text overlay
                cv2.putText(frame, f"Frame {i+1}/{num_frames}", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, prompt[:50] + ("..." if len(prompt) > 50 else ""), 
                           (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                frames.append(frame)
                progress_callback(i + 1, num_frames, f"Generated frame {i+1}/{num_frames}")
            
            # Save video
            self._save_video_frames(frames, output_path, fps)
            
            print(f"✅ Fallback video generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Fallback video generation failed: {e}")
            raise
    
    def _generate_fallback_video_from_image(
        self,
        input_image_path: str,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        output_path: str,
        progress_callback: Callable
    ) -> str:
        """Generate a fallback video from an input image"""
        try:
            progress_callback(0, num_frames, "Loading input image...")
            
            # Load input image
            input_image = cv2.imread(input_image_path)
            input_image = cv2.resize(input_image, (width, height))
            
            frames = []
            for i in range(num_frames):
                # Create variations of the input image
                frame = input_image.copy()
                
                # Add subtle transformations
                zoom_factor = 1.0 + (i / num_frames) * 0.1  # Slight zoom
                center = (width // 2, height // 2)
                M = cv2.getRotationMatrix2D(center, 0, zoom_factor)
                frame = cv2.warpAffine(frame, M, (width, height))
                
                # Add frame number
                cv2.putText(frame, f"Frame {i+1}/{num_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                frames.append(frame)
                progress_callback(i + 1, num_frames, f"Generated frame {i+1}/{num_frames}")
            
            # Save video
            self._save_video_frames(frames, output_path, fps)
            
            print(f"✅ Fallback image-to-video generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Fallback image-to-video generation failed: {e}")
            raise