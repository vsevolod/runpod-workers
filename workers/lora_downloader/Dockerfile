# CivitAI LoRA downloader — thin RunPod serverless CPU worker
# Dockerfile path: workers/lora_downloader/Dockerfile
# Build context: repository root

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY workers/lora_downloader/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY workers/lora_downloader/schemas.py \
     workers/lora_downloader/handler.py \
     workers/lora_downloader/download.py \
     workers/lora_downloader/test_input.json \
     workers/lora_downloader/NOTICE \
     /app/
COPY workers/lora_downloader/LICENSES /app/LICENSES

ENV LORA_DIR=/runpod-volume/krea2/loras \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO

CMD ["python", "-u", "/app/handler.py"]
