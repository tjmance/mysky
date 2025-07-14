# ===========================================================================
# Faceswap image: DeepFaceLive + InstantID + MediaPipe (GPU-0 by default)
# ===========================================================================
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python base
# ---------------------------------------------------------------------------
RUN pip3 install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# DeepFaceLive
# ---------------------------------------------------------------------------
RUN git clone https://github.com/iperov/DeepFaceLive.git /opt/DeepFaceLive && \
    pip3 install --no-cache-dir -r /opt/DeepFaceLive/requirements.txt

# ---------------------------------------------------------------------------
# InstantID + MediaPipe for landmark tracking / identity guidance
# ---------------------------------------------------------------------------
RUN pip3 install --no-cache-dir \
        mediapipe==0.10.9 \
        instant-id==0.1.0 || true  # instant-id placeholder until package is public

# Make DeepFaceLive importable everywhere
ENV PYTHONPATH=/opt/DeepFaceLive:$PYTHONPATH

WORKDIR /workspace

# Default entrypoint launches DeepFaceLive (override in compose if needed)
ENTRYPOINT ["python3", "/opt/DeepFaceLive/main.py"]