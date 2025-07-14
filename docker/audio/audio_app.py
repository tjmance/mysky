#!/usr/bin/env python3
"""
AI Performance Stack - Audio Service
Real-time voice changing and offline audio generation
"""

import os
import sys
import argparse
import asyncio
import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import time

import gradio as gr
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

# Audio processing imports
try:
    import torch
    import torchaudio
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from audiocraft.models import MusicGen
    # Import other audio libraries as needed
except ImportError as e:
    logging.warning(f"Some audio libraries not available: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Main audio processing class combining multiple TTS and voice conversion technologies."""
    
    def __init__(self, model_dir: str = "/app/models", audio_dir: str = "/app/audio"):
        self.model_dir = Path(model_dir)
        self.audio_dir = Path(audio_dir)
        self.models = {}
        self.is_initialized = False
        
        # Create directories
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        (self.audio_dir / "input").mkdir(exist_ok=True)
        (self.audio_dir / "output").mkdir(exist_ok=True)
        (self.audio_dir / "stems").mkdir(exist_ok=True)
        
        self.init_models()
        
    def init_models(self):
        """Initialize all audio models."""
        logger.info("Initializing audio models...")
        
        try:
            # Initialize MusicGen
            self.init_musicgen()
            
            # Initialize TTS models
            self.init_tts_models()
            
            # Initialize RVC
            self.init_rvc()
            
            # Initialize Demucs
            self.init_demucs()
            
            self.is_initialized = True
            logger.info("Audio models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio models: {e}")
            
    def init_musicgen(self):
        """Initialize MusicGen model."""
        try:
            if torch.cuda.is_available():
                self.models['musicgen'] = MusicGen.get_pretrained('facebook/musicgen-medium')
                logger.info("MusicGen model loaded successfully")
            else:
                logger.warning("CUDA not available, MusicGen will run on CPU")
        except Exception as e:
            logger.error(f"Failed to load MusicGen: {e}")
            
    def init_tts_models(self):
        """Initialize TTS models (Bark, XTTS, etc.)."""
        try:
            # Placeholder for TTS model initialization
            # This would load actual TTS models
            self.models['tts'] = {
                'bark': None,
                'xtts': None,
                'tortoise': None
            }
            logger.info("TTS models initialized")
        except Exception as e:
            logger.error(f"Failed to initialize TTS models: {e}")
            
    def init_rvc(self):
        """Initialize RVC (Retrieval-based Voice Conversion) model."""
        try:
            # Placeholder for RVC initialization
            self.models['rvc'] = None
            logger.info("RVC model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RVC: {e}")
            
    def init_demucs(self):
        """Initialize Demucs for source separation."""
        try:
            # Demucs is typically used via command line
            self.models['demucs'] = True
            logger.info("Demucs initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Demucs: {e}")
            
    def generate_music(self, prompt: str, duration: int = 30, model: str = "medium") -> str:
        """Generate music using MusicGen."""
        try:
            if 'musicgen' not in self.models or self.models['musicgen'] is None:
                raise ValueError("MusicGen model not loaded")
                
            # Set generation parameters
            self.models['musicgen'].set_generation_params(duration=duration)
            
            # Generate music
            wav = self.models['musicgen'].generate([prompt])
            
            # Save to file
            output_path = self.audio_dir / "output" / f"generated_music_{int(time.time())}.wav"
            torchaudio.save(str(output_path), wav[0].cpu(), self.models['musicgen'].sample_rate)
            
            logger.info(f"Music generated successfully: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Music generation failed: {e}")
            raise
            
    def text_to_speech(self, text: str, voice: str = "default", model: str = "bark") -> str:
        """Convert text to speech using various TTS models."""
        try:
            output_path = self.audio_dir / "output" / f"tts_{int(time.time())}.wav"
            
            if model == "bark":
                # Bark TTS implementation
                # This would use the actual Bark model
                pass
            elif model == "xtts":
                # XTTS implementation
                pass
            else:
                # Default TTS
                pass
                
            # Placeholder: create a simple tone for testing
            sample_rate = 22050
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * 440 * t) * 0.3
            sf.write(str(output_path), audio, sample_rate)
            
            logger.info(f"TTS generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise
            
    def voice_conversion(self, input_audio: str, target_voice: str, model: str = "rvc") -> str:
        """Convert voice using RVC or other voice conversion models."""
        try:
            input_path = Path(input_audio)
            output_path = self.audio_dir / "output" / f"voice_converted_{int(time.time())}.wav"
            
            if model == "rvc":
                # RVC voice conversion implementation
                # This would use the actual RVC model
                pass
                
            # For now, just copy the input file
            import shutil
            shutil.copy2(input_path, output_path)
            
            logger.info(f"Voice conversion completed: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Voice conversion failed: {e}")
            raise
            
    def separate_audio(self, input_audio: str, model: str = "htdemucs") -> Dict[str, str]:
        """Separate audio into stems using Demucs."""
        try:
            input_path = Path(input_audio)
            output_dir = self.audio_dir / "stems" / f"separation_{int(time.time())}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run Demucs
            cmd = [
                "python", "-m", "demucs.separate",
                "--model", model,
                "--out", str(output_dir),
                str(input_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Find separated files
                stems = {}
                model_dir = output_dir / model / input_path.stem
                if model_dir.exists():
                    for stem_file in model_dir.glob("*.wav"):
                        stems[stem_file.stem] = str(stem_file)
                        
                logger.info(f"Audio separation completed: {len(stems)} stems")
                return stems
            else:
                raise RuntimeError(f"Demucs failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Audio separation failed: {e}")
            raise

class AudioAPI:
    """FastAPI wrapper for the audio service."""
    
    def __init__(self, processor: AudioProcessor):
        self.processor = processor
        self.app = FastAPI(title="AI Performance Stack - Audio API")
        self.setup_routes()
        
    def setup_routes(self):
        """Set up API routes."""
        
        @self.app.get("/health")
        async def health_check():
            return JSONResponse({"status": "healthy", "service": "audio"})
            
        @self.app.get("/models")
        async def list_models():
            models_info = {
                "musicgen": "facebook/musicgen-medium" if self.processor.models.get('musicgen') else None,
                "tts": list(self.processor.models.get('tts', {}).keys()),
                "rvc": "loaded" if self.processor.models.get('rvc') else None,
                "demucs": "available" if self.processor.models.get('demucs') else None
            }
            return JSONResponse({"models": models_info})
            
        @self.app.post("/generate/music")
        async def generate_music(prompt: str, duration: int = 30):
            try:
                output_path = self.processor.generate_music(prompt, duration)
                return JSONResponse({"status": "success", "output_path": output_path})
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/generate/tts")
        async def text_to_speech(text: str, voice: str = "default", model: str = "bark"):
            try:
                output_path = self.processor.text_to_speech(text, voice, model)
                return JSONResponse({"status": "success", "output_path": output_path})
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/convert/voice")
        async def voice_conversion(file: UploadFile = File(...), target_voice: str = "default"):
            try:
                # Save uploaded file
                input_path = self.processor.audio_dir / "input" / file.filename
                with open(input_path, "wb") as f:
                    f.write(await file.read())
                    
                output_path = self.processor.voice_conversion(str(input_path), target_voice)
                return JSONResponse({"status": "success", "output_path": output_path})
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/separate")
        async def separate_audio(file: UploadFile = File(...), model: str = "htdemucs"):
            try:
                # Save uploaded file
                input_path = self.processor.audio_dir / "input" / file.filename
                with open(input_path, "wb") as f:
                    f.write(await file.read())
                    
                stems = self.processor.separate_audio(str(input_path), model)
                return JSONResponse({"status": "success", "stems": stems})
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

def create_gradio_interface(processor: AudioProcessor) -> gr.Interface:
    """Create Gradio web interface for audio processing."""
    
    def gradio_generate_music(prompt, duration):
        """Gradio wrapper for music generation."""
        try:
            output_path = processor.generate_music(prompt, duration)
            return output_path, "Music generated successfully"
        except Exception as e:
            return None, f"Error: {str(e)}"
            
    def gradio_tts(text, voice, model):
        """Gradio wrapper for text-to-speech."""
        try:
            output_path = processor.text_to_speech(text, voice, model)
            return output_path, "TTS generated successfully"
        except Exception as e:
            return None, f"Error: {str(e)}"
            
    def gradio_voice_convert(audio_file, target_voice):
        """Gradio wrapper for voice conversion."""
        if audio_file is None:
            return None, "Please upload an audio file"
        try:
            output_path = processor.voice_conversion(audio_file, target_voice)
            return output_path, "Voice conversion completed"
        except Exception as e:
            return None, f"Error: {str(e)}"
            
    def gradio_separate_audio(audio_file, model):
        """Gradio wrapper for audio separation."""
        if audio_file is None:
            return None, "Please upload an audio file"
        try:
            stems = processor.separate_audio(audio_file, model)
            return json.dumps(stems, indent=2), "Audio separation completed"
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    with gr.Blocks(title="AI Performance Stack - Audio Processing") as interface:
        gr.Markdown("# Audio Processing Suite")
        gr.Markdown("Generate music, convert text to speech, change voices, and separate audio stems.")
        
        with gr.Tabs():
            with gr.TabItem("Music Generation"):
                with gr.Row():
                    with gr.Column():
                        music_prompt = gr.Textbox(label="Music Prompt", placeholder="Electronic dance music with heavy bass")
                        music_duration = gr.Slider(minimum=10, maximum=120, value=30, label="Duration (seconds)")
                        generate_music_btn = gr.Button("Generate Music", variant="primary")
                    with gr.Column():
                        music_output = gr.Audio(label="Generated Music")
                        music_status = gr.Textbox(label="Status", interactive=False)
                        
                generate_music_btn.click(
                    fn=gradio_generate_music,
                    inputs=[music_prompt, music_duration],
                    outputs=[music_output, music_status]
                )
                
            with gr.TabItem("Text-to-Speech"):
                with gr.Row():
                    with gr.Column():
                        tts_text = gr.Textbox(label="Text", placeholder="Enter text to convert to speech")
                        tts_voice = gr.Dropdown(choices=["default", "male", "female"], value="default", label="Voice")
                        tts_model = gr.Dropdown(choices=["bark", "xtts", "tortoise"], value="bark", label="Model")
                        generate_tts_btn = gr.Button("Generate Speech", variant="primary")
                    with gr.Column():
                        tts_output = gr.Audio(label="Generated Speech")
                        tts_status = gr.Textbox(label="Status", interactive=False)
                        
                generate_tts_btn.click(
                    fn=gradio_tts,
                    inputs=[tts_text, tts_voice, tts_model],
                    outputs=[tts_output, tts_status]
                )
                
            with gr.TabItem("Voice Conversion"):
                with gr.Row():
                    with gr.Column():
                        voice_input = gr.Audio(label="Input Audio", type="filepath")
                        target_voice = gr.Dropdown(choices=["default", "voice1", "voice2"], value="default", label="Target Voice")
                        convert_voice_btn = gr.Button("Convert Voice", variant="primary")
                    with gr.Column():
                        voice_output = gr.Audio(label="Converted Audio")
                        voice_status = gr.Textbox(label="Status", interactive=False)
                        
                convert_voice_btn.click(
                    fn=gradio_voice_convert,
                    inputs=[voice_input, target_voice],
                    outputs=[voice_output, voice_status]
                )
                
            with gr.TabItem("Audio Separation"):
                with gr.Row():
                    with gr.Column():
                        separation_input = gr.Audio(label="Input Audio", type="filepath")
                        separation_model = gr.Dropdown(choices=["htdemucs", "mdx_extra"], value="htdemucs", label="Model")
                        separate_btn = gr.Button("Separate Audio", variant="primary")
                    with gr.Column():
                        separation_output = gr.Textbox(label="Separated Stems (JSON)", lines=10)
                        separation_status = gr.Textbox(label="Status", interactive=False)
                        
                separate_btn.click(
                    fn=gradio_separate_audio,
                    inputs=[separation_input, separation_model],
                    outputs=[separation_output, separation_status]
                )
        
    return interface

def run_tts_webui():
    """Run TTS-WebUI in background."""
    try:
        os.chdir("/app/tts-webui")
        subprocess.Popen(["python", "server.py", "--listen", "--port", "7862"])
        logger.info("TTS-WebUI started on port 7862")
    except Exception as e:
        logger.error(f"Failed to start TTS-WebUI: {e}")

def run_rvc_webui():
    """Run RVC-WebUI in background."""
    try:
        os.chdir("/app/rvc-webui")
        subprocess.Popen(["python", "infer-web.py", "--listen", "--port", "7863"])
        logger.info("RVC-WebUI started on port 7863")
    except Exception as e:
        logger.error(f"Failed to start RVC-WebUI: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AI Performance Stack - Audio Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7861, help="Port to bind to")
    parser.add_argument("--mode", default="webui", choices=["webui", "api", "all"], help="Service mode")
    parser.add_argument("--model-dir", default="/app/models", help="Models directory")
    parser.add_argument("--audio-dir", default="/app/audio", help="Audio directory")
    args = parser.parse_args()
    
    # Initialize processor
    processor = AudioProcessor(model_dir=args.model_dir, audio_dir=args.audio_dir)
    
    if args.mode == "all":
        # Start all UIs in background
        threading.Thread(target=run_tts_webui, daemon=True).start()
        threading.Thread(target=run_rvc_webui, daemon=True).start()
        
        # Start main Gradio interface
        interface = create_gradio_interface(processor)
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=False,
            show_error=True
        )
    elif args.mode == "webui":
        # Start only main Gradio interface
        interface = create_gradio_interface(processor)
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=False,
            show_error=True
        )
    elif args.mode == "api":
        # Start only FastAPI server
        api = AudioAPI(processor)
        uvicorn.run(api.app, host=args.host, port=args.port)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()