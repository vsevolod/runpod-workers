# MiniMax H3 T2V — thin RunPod serverless worker
# Weights live on Network Volume (MODEL_DIR), not in this image (~144 GB t2va).
#
# RunPod GitHub Integration:
#   Dockerfile path:  workers/minimax_h3/Dockerfile
#   Docker build context: repository root (default)
#
# Local build from monorepo root:
#   docker build -f workers/minimax_h3/Dockerfile -t runpod-minimax-h3 .

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    ca-certificates \
    git \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3

RUN python -m pip install --no-cache-dir uv

ENV PATH="/.venv/bin:${PATH}"
RUN uv venv --python 3.12 /.venv

# CUDA torch (cu124 matches this base image).
RUN uv pip install --no-cache \
    "torch==2.7.0" "torchvision==0.22.0" \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    || uv pip install --no-cache torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu124

WORKDIR /app

COPY workers/minimax_h3/requirements.txt /app/requirements.txt
RUN uv pip install --no-cache -r /app/requirements.txt

COPY workers/minimax_h3/ /app/

# Do NOT set HF_HUB_ENABLE_HF_TRANSFER (not used; optional HF_XET on bootstrap pod).
ENV MODEL_DIR=/runpod-volume/minimax_h3 \
    MODEL_ID=MiniMaxAI/MiniMax-H3 \
    PYTHONPATH=/app \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# No weight download at build time — mount Network Volume at runtime.
CMD ["python", "-u", "/app/handler.py"]
