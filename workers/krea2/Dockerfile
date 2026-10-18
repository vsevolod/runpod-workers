# Krea 2 Turbo (FP8 default / INT8 ConvRot via DIT_QUANT) — thin RunPod serverless worker
# Weights download to container disk at worker startup, not into this image.
#
# RunPod GitHub Integration:
#   Dockerfile path:  workers/krea2/Dockerfile
#   Docker build context: repository root (default)
#
# Local build from monorepo root:
#   docker build -f workers/krea2/Dockerfile -t runpod-krea2 .

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:17e2934e1fa96152b14f78078bfbafd0f00f391df995dc6c641a720fce1202bb

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

# Use the GPU-calibrated pair and constrain subsequent dependency installs too.
COPY workers/shared/torch-cu128.constraints.txt /app/torch-constraints.txt
RUN uv pip install --no-cache -r /app/torch-constraints.txt \
    --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app

COPY workers/krea2/requirements.txt /app/requirements.txt
RUN uv pip install --no-cache -c /app/torch-constraints.txt -r /app/requirements.txt

COPY workers/krea2/schemas.py \
     workers/krea2/handler.py \
     workers/krea2/download_weights.py \
     workers/krea2/start.sh \
     workers/krea2/test_input.json \
     /app/
COPY workers/krea2/krea2_infer /app/krea2_infer
COPY workers/krea2/NOTICE /app/NOTICE
COPY workers/krea2/LICENSES /app/LICENSES
COPY workers/shared /app/shared

ENV MODEL_DIR=/models/krea2 \
    HF_HOME=/models/hf \
    LORA_DIR=/runpod-volume/krea2/loras \
    PYTHONPATH=/app \
    TORCH_COMPILE_DISABLE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Download in the background; load each component after its local files are ready.
CMD ["sh", "/app/start.sh"]
