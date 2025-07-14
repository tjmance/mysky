# AI Performance Stack

A mono-repository inside the project providing two GPU-accelerated Docker images (_faceswap_ and _audio_) plus a single Compose file that lets you spin up just what you need—studio-only audio work or the full real-time stream rig—without touching your host OS.

---

## Directory layout

```
ai-performance-stack/
├── compose/
│   └── docker-compose.yml   # One file rules them all
├── docker/
│   ├── faceswap/            # DeepFaceLive (+InstantID) image
│   │   └── Dockerfile
│   └── audio/               # TTS-WebUI bundle image
│       └── Dockerfile
└── docs/
    └── ai-performance-stack.md  # ← you are here
```

Large artefacts live **next to** the repo so `git pull` stays fast:

```
../models/   # DeepFaceLive, InstantID, RVC, MusicGen, …      (read-only)
../audio/    # Raw clips, stem exports, final tracks          (read-write)
```

---

## Quick start (Ubuntu 22.04 + NVIDIA driver)

```bash
# 1. Install Docker & NVIDIA runtime
sudo apt install docker docker-compose-plugin
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
. /etc/os-release && curl -s -L "https://nvidia.github.io/nvidia-docker/$ID$VERSION_ID/nvidia-docker.list" | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-docker2
sudo systemctl restart docker

# 2. Build images (from repo root)
docker compose -f compose/docker-compose.yml build

# 3a. Audio-only studio (GPU 1)
docker compose -f compose/docker-compose.yml up -d audio

# 3b. Full stream rig (Faceswap + Tracker on GPU 0, Audio on GPU 1)
docker compose -f compose/docker-compose.yml up -d --profile stream

# 4. Stop everything
docker compose -f compose/docker-compose.yml down
```

---

## Hot-swapping models
Drop new checkpoints into `../models/`—containers read from the mounted volume at runtime, so **no rebuild needed**.

---

## GPU assignment
Every service sets `CUDA_VISIBLE_DEVICES` so nothing steps on another process:

| Service   | GPU | Profile  |
|-----------|-----|----------|
| faceswap  | 0   | stream   |
| tracker   | 0   | stream   |
| audio     | 1   | _always_ |

If you have only one GPU, change both `0` and `1` to `0` in `compose/docker-compose.yml`.

---

## Updating

```bash
git pull
docker compose -f compose/docker-compose.yml pull --ignore-buildable
```

---

## Roadmap
1. Pre-built images on GHCR for one-command deployment.
2. Optional OBS WebSocket relay for automatic scene switching.
3. Prometheus & Grafana stack for real-time GPU monitoring.