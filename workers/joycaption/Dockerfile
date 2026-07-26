# JoyCaption Beta One — thin RunPod serverless worker
# Weights live on Network Volume (MODEL_DIR), not in this image.
#
# RunPod GitHub Integration:
#   Dockerfile path:  workers/joycaption/Dockerfile
#   Docker build context: repository root (default)
#
# Local build from monorepo root:
#   docker build -f workers/joycaption/Dockerfile -t runpod-joycaption .
#
# Known-working pins (JoyCaption upstream + CUDA image):
#   torch/torchvision from cu124 index (target 2.7.x when available)
#   transformers==4.51.3, accelerate==1.6.0, pillow==11.2.1

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

# CUDA torch (cu124 matches this base image). Prefer 2.7.x (JoyCaption known-working).
RUN uv pip install --no-cache \
    "torch==2.7.0" "torchvision==0.22.0" \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    || uv pip install --no-cache torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu124

WORKDIR /app

COPY workers/joycaption/requirements.txt /app/requirements.txt
RUN uv pip install --no-cache -r /app/requirements.txt

COPY workers/joycaption/schemas.py \
     workers/joycaption/handler.py \
     workers/joycaption/caption_utils.py \
     workers/joycaption/download_weights.py \
     workers/joycaption/test_input.json \
     workers/joycaption/NOTICE \
     /app/
COPY workers/joycaption/LICENSES /app/LICENSES

ENV MODEL_DIR=/runpod-volume/joycaption \
    MODEL_ID=fancyfeast/llama-joycaption-beta-one-hf-llava \
    PYTHONPATH=/app \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# No weight download at build time — mount Network Volume at runtime.
CMD ["python", "-u", "/app/handler.py"]
