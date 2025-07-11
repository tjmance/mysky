#!/usr/bin/env python3
"""
Test script for AI Video Generation Studio
Verifies system installation and basic functionality
"""

import sys
import os
import tempfile
import traceback
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    tests = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("streamlit", "Streamlit"),
        ("psutil", "psutil"),
        ("yaml", "PyYAML")
    ]
    
    failed = []
    for module, name in tests:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed.append(name)
    
    return len(failed) == 0

def test_cuda():
    """Test CUDA availability"""
    print("\n🔥 Testing CUDA...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✅ CUDA available with {device_count} device(s)")
            
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                memory_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {name} ({memory_gb:.1f} GB)")
            
            return True
        else:
            print("⚠️ CUDA not available - will use CPU mode")
            return False
            
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

def test_backend_imports():
    """Test backend module imports"""
    print("\n🔧 Testing backend modules...")
    
    try:
        # Add current directory to path for imports
        sys.path.insert(0, ".")
        
        from backend.utils import setup_environment, get_device_info, validate_inputs
        print("✅ Backend utils")
        
        from backend.video_generator import VideoGenerator
        print("✅ Video generator")
        
        from backend.skyreels_integration import SkyReelsGenerator
        print("✅ SkyReels integration")
        
        return True
        
    except Exception as e:
        print(f"❌ Backend import failed: {e}")
        traceback.print_exc()
        return False

def test_directories():
    """Test directory structure"""
    print("\n📁 Testing directories...")
    
    required_dirs = ["outputs", "models", "temp", "uploads"]
    
    for directory in required_dirs:
        path = Path(directory)
        if path.exists():
            print(f"✅ {directory}/")
        else:
            print(f"⚠️ {directory}/ missing - creating...")
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ {directory}/ created")
    
    return True

def test_video_generator():
    """Test video generator initialization"""
    print("\n🎬 Testing video generator...")
    
    try:
        sys.path.insert(0, ".")
        from backend.video_generator import VideoGenerator
        
        # Test initialization
        generator = VideoGenerator()
        print("✅ VideoGenerator initialized")
        
        # Test system info
        info = generator.get_system_info()
        print(f"✅ System info: {info['setup_complete']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Video generator test failed: {e}")
        traceback.print_exc()
        return False

def test_fallback_generation():
    """Test fallback video generation"""
    print("\n🎞️ Testing fallback generation...")
    
    try:
        sys.path.insert(0, ".")
        from backend.skyreels_integration import SkyReelsGenerator
        import uuid
        
        # Create test generator
        generator = SkyReelsGenerator()
        
        # Test parameters
        test_params = {
            "prompt": "Test video generation",
            "negative_prompt": "",
            "mode": "Text-to-Video",
            "resolution": "540p (960x540)",
            "num_frames": 16,
            "fps": 12,
            "guidance_scale": 7.5,
            "num_inference_steps": 10,
            "seed": 42
        }
        
        # Generate test video
        output_path = f"test_output_{uuid.uuid4().hex[:8]}.mp4"
        
        def progress_callback(step, total, message):
            print(f"  Progress: {step}/{total} - {message}")
        
        result_path = generator.generate(
            **test_params,
            output_path=output_path,
            progress_callback=progress_callback
        )
        
        if Path(result_path).exists():
            file_size = Path(result_path).stat().st_size
            print(f"✅ Test video generated: {result_path} ({file_size} bytes)")
            
            # Clean up test file
            Path(result_path).unlink()
            print("✅ Test file cleaned up")
            
            return True
        else:
            print(f"❌ Test video not found: {result_path}")
            return False
            
    except Exception as e:
        print(f"❌ Fallback generation test failed: {e}")
        traceback.print_exc()
        return False

def test_streamlit_app():
    """Test Streamlit app syntax"""
    print("\n🖥️ Testing Streamlit app...")
    
    try:
        # Test app import/syntax by compiling
        with open("app.py", 'r') as f:
            app_code = f.read()
        
        compile(app_code, "app.py", "exec")
        print("✅ Streamlit app syntax valid")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Streamlit app syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Streamlit app test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("🧪 AI Video Generation Studio - System Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("CUDA Test", test_cuda),
        ("Directory Test", test_directories),
        ("Backend Import Test", test_backend_imports),
        ("Video Generator Test", test_video_generator),
        ("Fallback Generation Test", test_fallback_generation),
        ("Streamlit App Test", test_streamlit_app)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! System is ready.")
        print("\nTo start the application:")
        print("  streamlit run app.py")
        return True
    else:
        print(f"\n⚠️ {len(results) - passed} tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)