#!/bin/sh
set -eu

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export MODEL_DIR="${MODEL_DIR:-/models/qwen_image_21_turbo}"
export HF_HOME="${HF_HOME:-/models/hf}"

# snapshot_download resumes interrupted downloads and checks existing files.
# A partial snapshot must never be handed to the inference process.
python -u "$APP_DIR/download_weights.py"
export LOCAL_FILES_ONLY=1
exec python -u "$APP_DIR/handler.py"
