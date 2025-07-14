# ===========================================================================
# Audio image: TTS-WebUI bundle (Bark, XTTS, MusicGen, YuE, RVC, Demucs)
# Uses GPU-1 by default (configured via CUDA_VISIBLE_DEVICES at runtime)
# ===========================================================================
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip git ffmpeg sox libsox-fmt-all && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python base
# ---------------------------------------------------------------------------
RUN pip3 install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# Clone TTS-WebUI (or similar) and install core requirements
# ---------------------------------------------------------------------------
RUN git clone https://github.com/voice-weaver/tts-webui.git /opt/tts-webui && \
    pip3 install --no-cache-dir -r /opt/tts-webui/requirements.txt

# ---------------------------------------------------------------------------
# Individual model toolkits
# ---------------------------------------------------------------------------
RUN pip3 install --no-cache-dir \
        bark-tokenizer \
        TTS==0.21.1 \
        audiocraft \
        demucs==4.0.2 \
        xformers \
        # Placeholder packages – replace with official PyPI names when available
        xtts==0.0.3 \
        rvc==0.0.1

WORKDIR /opt/tts-webui

EXPOSE 7860

# Launch Gradio / FastAPI host (override via compose if you have a different CLI)
ENTRYPOINT ["python3", "app.py", "--listen", "--port", "7860"]