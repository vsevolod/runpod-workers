# MiniMax H3 T2V — headless ComfyUI (native nodes) on RunPod serverless.
# Weights: RunPod Model Cache (HF hub layout) or legacy Network Volume (MODEL_DIR).
# Never baked into this image; materialize via model_store.py → /models (symlinks).
#
# RunPod GitHub Integration:
#   Dockerfile path:  workers/minimax_h3_comfy/Dockerfile
#   Docker build context: repository root
#
# ComfyUI pin is authoritative (see PINS.md). Do not assume worker-comfyui
# image tags include H3 support.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    COMFYUI_PATH=/comfyui \
    COMFY_URL=http://127.0.0.1:8188 \
    COMFY_OUTPUT_DIR=/comfyui/output \
    MODEL_NAME=Comfy-Org/MiniMax-H3 \
    HF_CACHE_ROOT=/runpod-volume/huggingface-cache/hub

# Pin: ComfyUI v0.30.0 (H3 native) — see workers/minimax_h3_comfy/PINS.md
ARG COMFYUI_REF=b1693ecba9f5b65f8c80ab36b195ab963ec92413

# build-essential: Triton (torch) compiles cuda_utils at runtime and needs gcc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    ca-certificates \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc \
    CXX=g++

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3

RUN python -m pip install --no-cache-dir uv

ENV PATH="/.venv/bin:${PATH}"
RUN uv venv --python 3.12 /.venv

# CUDA torch (cu124). uv's default first-index strategy stops at PyPI once
# it sees `torch` there (no cp312+cu124 wheel) and also cannot pull
# nvidia-cudnn-cu12==9.1.0.70 from PyPI if the cu124 index listed first.
# Both indexes are trusted here.
RUN uv pip install --no-cache torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --index-strategy unsafe-best-match

# Pinned ComfyUI (no custom-node packs for v1)
RUN git clone https://github.com/Comfy-Org/ComfyUI.git "${COMFYUI_PATH}" \
    && cd "${COMFYUI_PATH}" \
    && git checkout "${COMFYUI_REF}" \
    && uv pip install --no-cache -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        --index-strategy unsafe-best-match

WORKDIR /app

COPY workers/minimax_h3_comfy/requirements.txt /app/requirements.txt
RUN uv pip install --no-cache -r /app/requirements.txt

COPY workers/minimax_h3_comfy/handler.py \
     workers/minimax_h3_comfy/workflow.py \
     workers/minimax_h3_comfy/image_input.py \
     workers/minimax_h3_comfy/download_weights.py \
     workers/minimax_h3_comfy/model_store.py \
     workers/minimax_h3_comfy/test_input.json \
     workers/minimax_h3_comfy/start.sh \
     workers/minimax_h3_comfy/entrypoint.sh \
     workers/minimax_h3_comfy/PINS.md \
     /app/
COPY workers/minimax_h3_comfy/workflows /app/workflows

RUN chmod +x /app/start.sh /app/entrypoint.sh \
    && sed -i 's/\r$//' /app/start.sh /app/entrypoint.sh \
    && head -1 /app/entrypoint.sh \
    && test -x /app/entrypoint.sh

# Handler imports workflow from /app; template path is next to workflow.py
ENV PYTHONPATH=/app \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONUNBUFFERED=1

# Do NOT override "Docker Command" in RunPod UI — leave empty to use this CMD.
# entrypoint always prints to logs; on failure sleeps so UI can scrape output.
CMD ["/bin/bash", "/app/entrypoint.sh"]
