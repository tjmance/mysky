import streamlit as st
import os
import time
import subprocess
import json
from pathlib import Path
from PIL import Image
import tempfile
import shutil
from typing import Optional, Dict, Any
import uuid

from backend.video_generator import VideoGenerator
from backend.utils import get_available_models, validate_inputs, format_duration
from backend.upscaler import VideoUpscaler
from backend.frame_interpolator import FrameInterpolator
from backend.multi_gpu import multi_gpu_manager

# Page configuration
st.set_page_config(
    page_title="AI Video Generation Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .generation-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-processing {
        color: #007bff;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    if 'current_job' not in st.session_state:
        st.session_state.current_job = None
    if 'video_generator' not in st.session_state:
        st.session_state.video_generator = VideoGenerator()
    if 'upscaler' not in st.session_state:
        st.session_state.upscaler = VideoUpscaler()
    if 'frame_interpolator' not in st.session_state:
        st.session_state.frame_interpolator = FrameInterpolator()

def main():
    """Main application function"""
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 AI Video Generation Studio</h1>
        <p>Powered by SkyReels v2 - Local NSFW-Free Video Generation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Generation Settings")
        
        # Model selection
        available_models = get_available_models()
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            index=0 if available_models else None,
            help="Choose the video generation model"
        )
        
        # Generation mode
        generation_mode = st.selectbox(
            "Generation Mode",
            ["Text-to-Video", "Image-to-Video", "Video Extension"],
            help="Select the type of video generation"
        )
        
        # Basic settings
        st.subheader("📐 Video Settings")
        
        resolution = st.selectbox(
            "Resolution",
            ["540p (960x540)", "720p (1280x720)"],
            help="Output video resolution"
        )
        
        num_frames = st.slider(
            "Number of Frames",
            min_value=16,
            max_value=128,
            value=64,
            step=8,
            help="Total number of frames to generate"
        )
        
        fps = st.slider(
            "FPS (Frames Per Second)",
            min_value=8,
            max_value=30,
            value=24,
            step=2,
            help="Video playback speed"
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            seed = st.number_input(
                "Seed (-1 for random)",
                min_value=-1,
                max_value=2147483647,
                value=-1,
                help="Random seed for reproducible generation"
            )
            
            guidance_scale = st.slider(
                "Guidance Scale",
                min_value=1.0,
                max_value=20.0,
                value=7.5,
                step=0.5,
                help="Higher values follow prompt more strictly"
            )
            
            num_inference_steps = st.slider(
                "Inference Steps",
                min_value=10,
                max_value=50,
                value=25,
                step=5,
                help="More steps = better quality but slower generation"
            )
        
        # Enhanced Processing Options
        with st.expander("🚀 Enhanced Processing"):
            st.subheader("🔍 Video Upscaling")
            enable_upscaling = st.checkbox(
                "Enable Real-ESRGAN Upscaling",
                help="Upscale generated videos to higher resolutions"
            )
            
            if enable_upscaling:
                upscale_models = st.session_state.upscaler.get_available_models()
                upscale_model = st.selectbox(
                    "Upscaling Model",
                    list(upscale_models.keys()),
                    help="Choose upscaling model type"
                )
                
                upscale_factor = st.selectbox(
                    "Upscale Factor",
                    [2, 4],
                    index=1,
                    help="How much to upscale the video"
                )
            
            st.subheader("🎞️ Frame Interpolation") 
            enable_interpolation = st.checkbox(
                "Enable RIFE Frame Interpolation",
                help="Create smoother videos with higher FPS"
            )
            
            if enable_interpolation:
                target_fps = st.slider(
                    "Target FPS",
                    min_value=30,
                    max_value=120,
                    value=60,
                    step=15,
                    help="Target frames per second for smooth motion"
                )
                
                interpolation_factor = st.selectbox(
                    "Interpolation Factor",
                    [2, 4, 8],
                    index=1,
                    help="How many frames to interpolate between existing frames"
                )
        
        # Multi-GPU Information
        with st.expander("🖥️ System Information"):
            gpu_info = multi_gpu_manager.get_system_info()
            
            if gpu_info["cuda_available"]:
                st.write(f"**GPUs Available:** {gpu_info['gpu_count']}")
                
                if gpu_info["gpu_count"] > 1:
                    st.write(f"**Multi-GPU Mode:** {'Distributed' if gpu_info['is_distributed'] else 'DataParallel'}")
                    
                    if gpu_info["gpu_memory"]:
                        for gpu_id, memory in gpu_info["gpu_memory"].items():
                            st.write(f"**GPU {gpu_id}:** {memory['utilization_percent']:.1f}% used ({memory['reserved_gb']:.1f}GB/{memory['total_gb']:.1f}GB)")
                else:
                    st.write("**Single GPU Mode**")
            else:
                st.warning("CUDA not available - using CPU mode")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎭 Create Your Video")
        
        # Text prompt (always required)
        prompt = st.text_area(
            "Describe your video",
            placeholder="A majestic eagle soaring over snow-capped mountains at sunrise...",
            height=100,
            help="Describe what you want to see in the video"
        )
        
        # Negative prompt
        negative_prompt = st.text_area(
            "Negative prompt (optional)",
            placeholder="blurry, low quality, distorted...",
            height=60,
            help="Describe what you don't want in the video"
        )
        
        # Conditional inputs based on generation mode
        uploaded_file = None
        if generation_mode == "Image-to-Video":
            st.subheader("📸 Upload Source Image")
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'webp'],
                help="Upload an image to use as the starting frame"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Source Image", use_column_width=True)
                
        elif generation_mode == "Video Extension":
            st.subheader("🎥 Upload Source Video")
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'mov', 'avi', 'mkv'],
                help="Upload a video to extend"
            )
            
            if uploaded_file:
                st.video(uploaded_file)
        
        # Generation button
        if st.button("🚀 Generate Video", type="primary", use_container_width=True):
            if not prompt.strip():
                st.error("Please enter a prompt!")
                return
            
            if generation_mode != "Text-to-Video" and not uploaded_file:
                st.error(f"Please upload a {'image' if generation_mode == 'Image-to-Video' else 'video'} file!")
                return
            
            # Prepare generation parameters
            generation_params = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "mode": generation_mode,
                "resolution": resolution,
                "num_frames": num_frames,
                "fps": fps,
                "seed": seed if seed != -1 else None,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "model": selected_model,
                # Enhanced processing options
                "enable_upscaling": enable_upscaling,
                "upscale_model": upscale_model if enable_upscaling else None,
                "upscale_factor": upscale_factor if enable_upscaling else None,
                "enable_interpolation": enable_interpolation,
                "target_fps": target_fps if enable_interpolation else None,
                "interpolation_factor": interpolation_factor if enable_interpolation else None
            }
            
            # Start generation
            generate_video(generation_params, uploaded_file)
    
    with col2:
        st.header("📊 Status & History")
        
        # Current job status
        if st.session_state.current_job:
            show_generation_status()
        
        # Generation history
        if st.session_state.generation_history:
            st.subheader("🎬 Recent Generations")
            for i, job in enumerate(reversed(st.session_state.generation_history[-5:])):
                show_generation_result(job, i)

def generate_video(params: Dict[str, Any], uploaded_file: Optional[st.runtime.uploaded_file_manager.UploadedFile]):
    """Start video generation process"""
    try:
        # Create unique job ID
        job_id = str(uuid.uuid4())
        
        # Prepare input file if provided
        input_file_path = None
        if uploaded_file:
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            file_extension = uploaded_file.name.split('.')[-1]
            input_file_path = temp_dir / f"{job_id}_input.{file_extension}"
            
            with open(input_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Update session state
        st.session_state.current_job = {
            "id": job_id,
            "status": "starting",
            "progress": 0,
            "params": params,
            "input_file": str(input_file_path) if input_file_path else None,
            "start_time": time.time(),
            "output_path": None,
            "error": None
        }
        
        # Start generation in background
        st.session_state.video_generator.generate_async(
            job_id=job_id,
            params=params,
            input_file=input_file_path
        )
        
        st.success("🎬 Video generation started!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Failed to start generation: {str(e)}")

def show_generation_status():
    """Display current generation status"""
    job = st.session_state.current_job
    
    if not job:
        return
    
    # Check job status
    status = st.session_state.video_generator.get_job_status(job["id"])
    
    if status:
        job.update(status)
    
    st.subheader("🔄 Current Generation")
    
    status_color = {
        "starting": "🟡",
        "running": "🔵",
        "completed": "🟢",
        "failed": "🔴"
    }
    
    st.write(f"{status_color.get(job['status'], '⚪')} **Status:** {job['status'].title()}")
    
    if job["status"] == "running" and job.get("progress", 0) > 0:
        progress = job["progress"]
        st.progress(progress / 100)
        st.write(f"Progress: {progress:.1f}%")
    
    # Show elapsed time
    elapsed = time.time() - job["start_time"]
    st.write(f"⏱️ **Elapsed:** {format_duration(elapsed)}")
    
    # Show estimated time remaining
    if job["status"] == "running" and job.get("progress", 0) > 0:
        progress = job["progress"]
        if progress > 5:  # Only show estimate after 5% progress
            estimated_total = elapsed / (progress / 100)
            remaining = estimated_total - elapsed
            st.write(f"⏳ **Estimated remaining:** {format_duration(remaining)}")
    
    # Show error if failed
    if job["status"] == "failed" and job.get("error"):
        st.error(f"Error: {job['error']}")
    
    # Show result if completed
    if job["status"] == "completed" and job.get("output_path"):
        output_path = Path(job["output_path"])
        if output_path.exists():
            st.success("✅ Generation completed!")
            st.video(str(output_path))
            
            # Move to history
            st.session_state.generation_history.append(job.copy())
            st.session_state.current_job = None
            
            # Auto-rerun to update UI
            st.rerun()
    
    # Auto-refresh for active jobs
    if job["status"] in ["starting", "running"]:
        time.sleep(2)
        st.rerun()

def show_generation_result(job: Dict[str, Any], index: int):
    """Display a completed generation result"""
    with st.expander(f"Generation {index + 1} - {job['params']['mode']}", expanded=index == 0):
        col1, col2 = st.columns([3, 2])
        
        with col1:
            if job.get("output_path") and Path(job["output_path"]).exists():
                st.video(job["output_path"])
            else:
                st.error("Video file not found")
        
        with col2:
            st.write(f"**Prompt:** {job['params']['prompt'][:100]}...")
            st.write(f"**Resolution:** {job['params']['resolution']}")
            st.write(f"**Frames:** {job['params']['num_frames']}")
            st.write(f"**FPS:** {job['params']['fps']}")
            
            generation_time = job.get("end_time", job["start_time"]) - job["start_time"]
            st.write(f"**Generation Time:** {format_duration(generation_time)}")

if __name__ == "__main__":
    main()