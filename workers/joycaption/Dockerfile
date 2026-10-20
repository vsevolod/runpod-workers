# Gliese Qwen3.5 9B — RunPod image captioning worker.
# The joycaption package/image name preserves the existing endpoint integration.
# Pinned weights are downloaded to container disk, not baked into this image.
#
# RunPod GitHub Integration:
#   Dockerfile path:  workers/joycaption/Dockerfile
#   Docker build context: repository root (default)
#
# Local build from monorepo root:
#   docker build -f workers/joycaption/Dockerfile -t runpod-joycaption .
#
# The shared CUDA Torch pair includes Blackwell sm_120 support.

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

# Constrain later dependency installation too; never fall back to another pair.
COPY workers/shared/torch-cu128.constraints.txt /app/torch-constraints.txt
RUN uv pip install --no-cache -r /app/torch-constraints.txt \
    --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app

COPY workers/joycaption/requirements.txt /app/requirements.txt
RUN uv pip install --no-cache -c /app/torch-constraints.txt -r /app/requirements.txt

COPY workers/joycaption/schemas.py \
     workers/joycaption/handler.py \
     workers/joycaption/caption_runtime.py \
     workers/joycaption/caption_utils.py \
     workers/joycaption/model_store.py \
     workers/joycaption/download_weights.py \
     workers/joycaption/test_input.json \
     workers/joycaption/NOTICE \
     /app/
COPY workers/joycaption/LICENSES /app/LICENSES
COPY workers/shared /app/workers/shared

ENV MODEL_DIR=/models/gliese_qwen35_9b \
    HF_HOME=/models/hf \
    CAPTION_MAX_PIXELS=1048576 \
    PYTHONPATH=/app:/app/workers \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CPU prefetch starts before heavy runtime imports; each shard has its own barrier.
CMD ["python", "-u", "/app/handler.py"]
