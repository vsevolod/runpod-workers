# Qwen-Image-2.1 Base/Turbo — one request-selected RunPod worker (Diffusers).
# Weights download to container disk at worker startup, not into this image.
#
# RunPod GitHub Integration:
#   Dockerfile path:  workers/qwen_image_21/Dockerfile
#   Docker build context: repository root (default)
#
# Local build from monorepo root:
#   docker build -f workers/qwen_image_21/Dockerfile -t runpod-qwen-image-21 .

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
# torchvision is required by the Qwen3-VL image processor.
COPY workers/shared/torch-cu128.constraints.txt /app/torch-constraints.txt
RUN uv pip install --no-cache -r /app/torch-constraints.txt \
    --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app

COPY workers/qwen_image_21/requirements.txt /app/requirements.txt
RUN uv pip install --no-cache -c /app/torch-constraints.txt -r /app/requirements.txt

# The pinned upstream pipeline couples reference sizes to the output canvas.
# Fail the build if this narrow patch no longer applies after a dependency update.
COPY workers/qwen_image_21/patches /app/patches
RUN cd /.venv/lib/python3.12/site-packages && \
    git apply --check /app/patches/diffusers-native-references.patch && \
    git apply /app/patches/diffusers-native-references.patch

COPY workers/qwen_image_21/schemas.py \
     workers/qwen_image_21/handler.py \
     workers/qwen_image_21/download_weights.py \
     workers/qwen_image_21/start.sh \
     workers/qwen_image_21/test_input.json \
     /app/
COPY workers/qwen_image_21/qwen_image_21_infer /app/qwen_image_21_infer
COPY workers/qwen_image_21/LICENSES /app/LICENSES
COPY workers/shared /app/shared

ENV MODEL_DIR=/models/qwen_image_21_turbo \
    MODEL_ID=Qwen/Qwen-Image-2.1-Turbo \
    BASE_MODEL_DIR=/models/qwen_image_21_base \
    QWEN_VAE_DIR=/models/qwen_image_21_shared/texture_fix_vae \
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
