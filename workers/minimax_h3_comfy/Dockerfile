# MiniMax H3 T2V — headless ComfyUI (native nodes) on RunPod serverless.
# Weights live on Network Volume (MODEL_DIR), not in this image.
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
    MODEL_DIR=/runpod-volume/minimax_h3_comfy

# Pin: ComfyUI v0.30.0 (H3 native) — see workers/minimax_h3_comfy/PINS.md
ARG COMFYUI_REF=b1693ecba9f5b65f8c80ab36b195ab963ec92413

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    ca-certificates \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
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

# CUDA torch (cu124)
RUN uv pip install --no-cache torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Pinned ComfyUI (no custom-node packs for v1)
RUN git clone https://github.com/Comfy-Org/ComfyUI.git "${COMFYUI_PATH}" \
    && cd "${COMFYUI_PATH}" \
    && git checkout "${COMFYUI_REF}" \
    && uv pip install --no-cache -r requirements.txt

WORKDIR /app

COPY workers/minimax_h3_comfy/requirements.txt /app/requirements.txt
RUN uv pip install --no-cache -r /app/requirements.txt

COPY workers/minimax_h3_comfy/handler.py \
     workers/minimax_h3_comfy/workflow.py \
     workers/minimax_h3_comfy/download_weights.py \
     workers/minimax_h3_comfy/test_input.json \
     workers/minimax_h3_comfy/start.sh \
     workers/minimax_h3_comfy/PINS.md \
     /app/
COPY workers/minimax_h3_comfy/workflows /app/workflows

RUN chmod +x /app/start.sh

# Handler imports workflow from /app; template path is next to workflow.py
ENV PYTHONPATH=/app \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD ["/app/start.sh"]
