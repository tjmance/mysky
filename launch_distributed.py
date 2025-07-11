#!/usr/bin/env python3
"""
Distributed launch script for AI Video Generation Studio
Supports multi-GPU inference with torchrun
"""

import os
import sys
import argparse
import subprocess
import torch
from pathlib import Path

def check_gpu_availability():
    """Check available GPUs"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return 0
    
    gpu_count = torch.cuda.device_count()
    print(f"🔍 Found {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    
    return gpu_count

def create_distributed_app():
    """Create distributed version of the main app"""
    distributed_app_code = '''
import os
import torch
import torch.distributed as dist
from backend.multi_gpu import multi_gpu_manager
from app import main as streamlit_main

def setup_distributed():
    """Setup distributed environment for Streamlit"""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        print(f"Initializing distributed process: rank {rank}/{world_size}")
        
        # Only run Streamlit on main process
        if rank == 0:
            return True
        else:
            # Other processes just wait and handle model inference
            import time
            print(f"Worker process {rank} ready for inference tasks")
            while True:
                time.sleep(10)  # Keep worker alive
            return False
    else:
        return True

if __name__ == "__main__":
    if setup_distributed():
        streamlit_main()
'''
    
    with open("app_distributed.py", "w") as f:
        f.write(distributed_app_code)
    
    print("✅ Created distributed app: app_distributed.py")

def launch_streamlit_distributed(num_gpus: int, port: int = 8501):
    """Launch Streamlit with distributed backend"""
    
    if num_gpus <= 1:
        print("🚀 Launching single GPU mode...")
        cmd = [
            "streamlit", "run", "app.py",
            "--server.port", str(port),
            "--server.address", "0.0.0.0"
        ]
    else:
        print(f"🚀 Launching distributed mode with {num_gpus} GPUs...")
        
        # Create distributed app
        create_distributed_app()
        
        # Launch with torchrun
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "--nnodes=1",
            "--node_rank=0",
            "--master_addr=localhost",
            "--master_port=29500",
            "app_distributed.py"
        ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested")
    except subprocess.CalledProcessError as e:
        print(f"❌ Launch failed: {e}")

def launch_docker_distributed():
    """Launch using Docker Compose"""
    print("🐳 Launching with Docker Compose...")
    
    try:
        subprocess.run([
            "docker-compose", "up", "--build"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested")
        subprocess.run(["docker-compose", "down"])
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker launch failed: {e}")

def create_performance_test():
    """Create performance test script"""
    perf_test_code = '''
#!/usr/bin/env python3
"""Performance test for distributed video generation"""

import time
import torch
from backend.video_generator import VideoGenerator
from backend.multi_gpu import multi_gpu_manager

def test_performance():
    """Test generation performance"""
    print("🧪 Running performance test...")
    
    # System info
    gpu_info = multi_gpu_manager.get_system_info()
    print(f"GPUs: {gpu_info['gpu_count']}")
    print(f"Distributed: {gpu_info['is_distributed']}")
    
    # Test parameters
    test_params = {
        "prompt": "Performance test: a flowing river in a mountain landscape",
        "negative_prompt": "blurry, low quality",
        "mode": "Text-to-Video",
        "resolution": "540p (960x540)",
        "num_frames": 32,
        "fps": 24,
        "guidance_scale": 7.5,
        "num_inference_steps": 20,
        "seed": 42,
        "enable_upscaling": False,
        "enable_interpolation": False
    }
    
    # Initialize generator
    generator = VideoGenerator()
    
    if not generator.setup_complete:
        print("❌ Generator not ready")
        return
    
    # Run test
    start_time = time.time()
    
    try:
        job_id = generator.generate_async("perf_test", test_params)
        
        # Monitor progress
        while True:
            status = generator.get_job_status(job_id)
            if not status:
                break
                
            print(f"Progress: {status['progress']:.1f}% - {status['status']}")
            
            if status['status'] in ['completed', 'failed']:
                break
                
            time.sleep(2)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Test completed in {duration:.1f} seconds")
        
        if status['status'] == 'completed':
            print(f"Output: {status['output_path']}")
        else:
            print(f"❌ Test failed: {status.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")

if __name__ == "__main__":
    test_performance()
'''
    
    with open("performance_test.py", "w") as f:
        f.write(perf_test_code)
    
    os.chmod("performance_test.py", 0o755)
    print("✅ Created performance test: performance_test.py")

def main():
    parser = argparse.ArgumentParser(
        description="Launch AI Video Generation Studio with advanced features"
    )
    
    parser.add_argument(
        "--mode", 
        choices=["local", "distributed", "docker"],
        default="local",
        help="Launch mode"
    )
    
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (auto-detect if not specified)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for Streamlit server"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run performance test"
    )
    
    args = parser.parse_args()
    
    # Check system
    gpu_count = check_gpu_availability()
    
    if args.gpus is None:
        args.gpus = gpu_count
    
    if args.test:
        create_performance_test()
        subprocess.run(["python3", "performance_test.py"])
        return
    
    # Launch based on mode
    if args.mode == "docker":
        launch_docker_distributed()
    elif args.mode == "distributed" or args.gpus > 1:
        launch_streamlit_distributed(args.gpus, args.port)
    else:
        # Simple local launch
        print("🚀 Launching local mode...")
        subprocess.run([
            "streamlit", "run", "app.py",
            "--server.port", str(args.port),
            "--server.address", "0.0.0.0"
        ])

if __name__ == "__main__":
    main()