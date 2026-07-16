# Krea 2 Turbo FP8 — thin RunPod serverless worker
# Weights live on Network Volume (MODEL_DIR), not in this image.
#
# RunPod GitHub Integration:
#   Dockerfile path:  workers/krea2/Dockerfile
#   Docker build context: repository root (default)
#
# Local build from monorepo root:
#   docker build -f workers/krea2/Dockerfile -t runpod-krea2 .

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

# CUDA torch (cu124 matches this base image)
RUN uv pip install --no-cache torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu124

WORKDIR /app

COPY workers/krea2/requirements.txt /app/requirements.txt
RUN uv pip install --no-cache -r /app/requirements.txt

COPY workers/krea2/schemas.py \
     workers/krea2/handler.py \
     workers/krea2/download_weights.py \
     workers/krea2/test_input.json \
     /app/
COPY workers/krea2/krea2_infer /app/krea2_infer

ENV MODEL_DIR=/runpod-volume/krea2 \
    PYTHONPATH=/app \
    TORCH_COMPILE_DISABLE=1

# No weight download at build time — mount Network Volume at runtime.
CMD ["python", "-u", "/app/handler.py"]
