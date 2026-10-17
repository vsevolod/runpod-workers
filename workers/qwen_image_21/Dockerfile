# Qwen-Image-2.1-Turbo — thin RunPod serverless worker (Diffusers).
# Weights download to container disk at worker startup, not into this image.
#
# RunPod GitHub Integration:
#   Dockerfile path:  workers/qwen_image_21/Dockerfile
#   Docker build context: repository root (default)
#
# Local build from monorepo root:
#   docker build -f workers/qwen_image_21/Dockerfile -t runpod-qwen-image-21 .

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    ca-certificates \
    git \
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

# CUDA torch (cu124 matches this base image). torchvision is required by the
# Qwen3-VL image processor.
RUN uv pip install --no-cache torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu124

WORKDIR /app

COPY workers/qwen_image_21/requirements.txt /app/requirements.txt
RUN uv pip install --no-cache -r /app/requirements.txt

COPY workers/qwen_image_21/schemas.py \
     workers/qwen_image_21/handler.py \
     workers/qwen_image_21/download_weights.py \
     workers/qwen_image_21/start.sh \
     workers/qwen_image_21/test_input.json \
     /app/
COPY workers/qwen_image_21/qwen_image_21_infer /app/qwen_image_21_infer
COPY workers/shared /app/shared

ENV MODEL_DIR=/models/qwen_image_21_turbo \
    MODEL_ID=Qwen/Qwen-Image-2.1-Turbo \
    HF_HOME=/models/hf \
    LORA_DIR=/runpod-volume/qwen_image_21_turbo/loras \
    QWEN_DISABLE_MMAP=0 \
    QWEN_VAE=texture-fix \
    OFFLOAD=cpu \
    PYTHONPATH=/app \
    TORCH_COMPILE_DISABLE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Overlap downloads with imports; load only the completed local snapshot.
CMD ["sh", "/app/start.sh"]
