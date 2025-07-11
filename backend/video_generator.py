import os
import sys
import subprocess
import threading
import time
import json
import torch
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import uuid
import traceback

from .skyreels_integration import SkyReelsGenerator
from .utils import setup_environment, get_device_info, validate_cuda
from .upscaler import VideoUpscaler
from .frame_interpolator import FrameInterpolator
from .multi_gpu import multi_gpu_manager

@dataclass
class GenerationJob:
    """Represents a video generation job"""
    id: str
    status: str  # starting, running, completed, failed
    progress: float
    params: Dict[str, Any]
    input_file: Optional[str]
    output_path: Optional[str]
    error: Optional[str]
    start_time: float
    end_time: Optional[float]

class VideoGenerator:
    """Main video generation manager that orchestrates SkyReels v2"""
    
    def __init__(self):
        self.jobs: Dict[str, GenerationJob] = {}
        self.skyreels_generator = None
        self.upscaler = None
        self.frame_interpolator = None
        self.device_info = get_device_info()
        self.setup_complete = False
        self._setup_lock = threading.Lock()
        
        # Initialize on first use
        self._initialize()
    
    def _initialize(self):
        """Initialize the video generator and models"""
        try:
            # Validate CUDA availability
            if not validate_cuda():
                raise RuntimeError("CUDA not available or H100 GPU not detected")
            
            # Setup environment
            setup_environment()
            
            # Initialize SkyReels generator
            self.skyreels_generator = SkyReelsGenerator()
            
            # Initialize advanced features
            try:
                self.upscaler = VideoUpscaler()
                print("✅ Video upscaler initialized")
            except Exception as e:
                print(f"⚠️ Video upscaler initialization failed: {e}")
                self.upscaler = None
            
            try:
                self.frame_interpolator = FrameInterpolator()
                print("✅ Frame interpolator initialized")
            except Exception as e:
                print(f"⚠️ Frame interpolator initialization failed: {e}")
                self.frame_interpolator = None
            
            # Optimize GPU memory
            multi_gpu_manager.optimize_memory()
            
            self.setup_complete = True
            print("✅ VideoGenerator with advanced features initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize VideoGenerator: {e}")
            traceback.print_exc()
            self.setup_complete = False
    
    def generate_async(self, job_id: str, params: Dict[str, Any], input_file: Optional[Path] = None):
        """Start asynchronous video generation"""
        if not self.setup_complete:
            raise RuntimeError("VideoGenerator not properly initialized")
        
        # Create job
        job = GenerationJob(
            id=job_id,
            status="starting",
            progress=0.0,
            params=params,
            input_file=str(input_file) if input_file else None,
            output_path=None,
            error=None,
            start_time=time.time(),
            end_time=None
        )
        
        self.jobs[job_id] = job
        
        # Start generation in background thread
        thread = threading.Thread(
            target=self._generate_video_thread,
            args=(job,),
            daemon=True
        )
        thread.start()
        
        return job_id
    
    def _generate_video_thread(self, job: GenerationJob):
        """Background thread for video generation"""
        try:
            job.status = "running"
            job.progress = 1.0
            
            # Prepare output path
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"{job.params['mode']}_{timestamp}_{job.id[:8]}.mp4"
            output_path = output_dir / output_filename
            
            # Progress callback
            def progress_callback(step: int, total_steps: int, message: str = ""):
                progress = (step / total_steps) * 100
                job.progress = min(progress, 99.0)  # Keep at 99% until complete
                print(f"Progress: {progress:.1f}% - {message}")
            
            # Generate video using SkyReels
            result_path = self.skyreels_generator.generate(
                prompt=job.params["prompt"],
                negative_prompt=job.params.get("negative_prompt", ""),
                mode=job.params["mode"],
                resolution=job.params["resolution"],
                num_frames=job.params["num_frames"],
                fps=job.params["fps"],
                guidance_scale=job.params["guidance_scale"],
                num_inference_steps=job.params["num_inference_steps"],
                seed=job.params.get("seed"),
                input_file=job.input_file,
                output_path=str(output_path),
                progress_callback=progress_callback
            )
            
            # Post-processing: Apply enhancements
            final_path = self._apply_post_processing(
                result_path, job.params, progress_callback
            )
            
            # Mark as completed
            job.status = "completed"
            job.progress = 100.0
            job.output_path = str(final_path)
            job.end_time = time.time()
            
            print(f"✅ Video generation completed: {final_path}")
            
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.end_time = time.time()
            print(f"❌ Video generation failed: {e}")
            traceback.print_exc()
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a generation job"""
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        return {
            "status": job.status,
            "progress": job.progress,
            "output_path": job.output_path,
            "error": job.error,
            "start_time": job.start_time,
            "end_time": job.end_time
        }
    
    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Get status of all jobs"""
        return [
            {
                "id": job.id,
                "status": job.status,
                "progress": job.progress,
                "params": job.params,
                "output_path": job.output_path,
                "error": job.error,
                "start_time": job.start_time,
                "end_time": job.end_time
            }
            for job in self.jobs.values()
        ]
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        job = self.jobs.get(job_id)
        if not job or job.status not in ["starting", "running"]:
            return False
        
        # For now, we'll just mark it as failed
        # In a more sophisticated implementation, we'd interrupt the generation process
        job.status = "failed"
        job.error = "Cancelled by user"
        job.end_time = time.time()
        
        return True
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old completed jobs"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        jobs_to_remove = []
        for job_id, job in self.jobs.items():
            if job.status in ["completed", "failed"]:
                age = current_time - job.start_time
                if age > max_age_seconds:
                    jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.jobs[job_id]
        
        return len(jobs_to_remove)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for diagnostics"""
        return {
            "setup_complete": self.setup_complete,
            "device_info": self.device_info,
            "active_jobs": len([j for j in self.jobs.values() if j.status in ["starting", "running"]]),
            "total_jobs": len(self.jobs),
            "cuda_available": torch.cuda.is_available(),
            "gpu_memory": self._get_gpu_memory_info() if torch.cuda.is_available() else None
        }
    
    def _apply_post_processing(
        self, 
        video_path: str, 
        params: Dict[str, Any], 
        progress_callback: Callable
    ) -> str:
        """Apply post-processing enhancements to generated video"""
        
        current_path = video_path
        
        try:
            # Apply frame interpolation first (smoother motion)
            if params.get("enable_interpolation", False) and self.frame_interpolator:
                progress_callback(80, 100, "Applying frame interpolation...")
                
                interpolated_path = video_path.replace(".mp4", "_interpolated.mp4")
                
                current_path = self.frame_interpolator.interpolate_video(
                    input_video_path=current_path,
                    output_video_path=interpolated_path,
                    target_fps=params.get("target_fps", 60),
                    interpolation_factor=params.get("interpolation_factor", 2),
                    progress_callback=lambda step, total, msg: progress_callback(
                        80 + (step / total) * 10, 100, f"Interpolation: {msg}"
                    )
                )
                
                print(f"✅ Frame interpolation applied: {current_path}")
            
            # Apply upscaling second (higher resolution)
            if params.get("enable_upscaling", False) and self.upscaler:
                progress_callback(90, 100, "Applying video upscaling...")
                
                upscaled_path = current_path.replace(".mp4", "_upscaled.mp4")
                
                # Switch to specified model if needed
                upscale_model = params.get("upscale_model")
                if upscale_model:
                    try:
                        self.upscaler.switch_model(upscale_model)
                    except Exception as e:
                        print(f"⚠️ Failed to switch upscale model: {e}")
                
                current_path = self.upscaler.upscale_video(
                    input_video_path=current_path,
                    output_video_path=upscaled_path,
                    scale_factor=params.get("upscale_factor", 4),
                    progress_callback=lambda step, total, msg: progress_callback(
                        90 + (step / total) * 10, 100, f"Upscaling: {msg}"
                    )
                )
                
                print(f"✅ Video upscaling applied: {current_path}")
            
            progress_callback(100, 100, "Post-processing complete!")
            return current_path
            
        except Exception as e:
            print(f"⚠️ Post-processing failed: {e}")
            traceback.print_exc()
            # Return original path if post-processing fails
            return video_path
    
    def _get_gpu_memory_info(self) -> Dict[str, float]:
        """Get GPU memory information"""
        if not torch.cuda.is_available():
            return {}
        
        try:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            return {
                "allocated_gb": memory_allocated,
                "reserved_gb": memory_reserved,
                "total_gb": memory_total,
                "free_gb": memory_total - memory_reserved
            }
        except Exception:
            return {}