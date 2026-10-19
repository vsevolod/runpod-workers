# One request-selected image worker. Checkpoints remain on container disk.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:17e2934e1fa96152b14f78078bfbafd0f00f391df995dc6c641a720fce1202bb

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl ca-certificates git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12
RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3
RUN python -m pip install --no-cache-dir uv
ENV PATH="/.venv/bin:${PATH}"
RUN uv venv --python 3.12 /.venv

WORKDIR /app
COPY workers/shared/torch-cu128.constraints.txt /app/torch-constraints.txt
RUN uv pip install --no-cache -r /app/torch-constraints.txt \
    --index-url https://download.pytorch.org/whl/cu128
COPY workers/image/requirements.txt /app/requirements.txt
RUN uv pip install --no-cache -c /app/torch-constraints.txt -r /app/requirements.txt

COPY workers/qwen_image_21/patches /app/patches
RUN cd /.venv/lib/python3.12/site-packages && \
    git apply --check /app/patches/diffusers-native-references.patch && \
    git apply /app/patches/diffusers-native-references.patch

# Preserve source layout so both model downloaders and test suites resolve paths.
COPY workers/shared /app/workers/shared
COPY workers/krea2 /app/workers/krea2
COPY workers/qwen_image_21 /app/workers/qwen_image_21
COPY workers/image /app/workers/image
COPY scripts /app/scripts

ENV PYTHONPATH=/app/workers:/app/workers/image:/app/workers/krea2:/app/workers/qwen_image_21 \
    KREA_MODEL_DIR=/models/krea2 \
    KREA_LORA_DIR=/runpod-volume/krea2/loras \
    KREA_DIT_QUANT=int8_convrot \
    QWEN_MODEL_DIR=/models/qwen_image_21_turbo \
    QWEN_BASE_MODEL_DIR=/models/qwen_image_21_base \
    QWEN_LORA_DIR=/runpod-volume/qwen_image_21_turbo/loras \
    QWEN_VAE_DIR=/models/qwen_image_21_shared/texture_fix_vae \
    QWEN_DISABLE_MMAP=0 \
    QWEN_VAE=texture-fix \
    OFFLOAD=cpu \
    HF_HOME=/models/hf \
    LORA_CACHE_DIR=/models/lora_cache \
    TORCH_COMPILE_DISABLE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD ["sh", "/app/workers/image/start.sh"]
