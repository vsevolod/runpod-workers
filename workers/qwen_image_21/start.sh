#!/bin/sh
set -eu

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export MODEL_DIR="${MODEL_DIR:-/models/qwen_image_21_turbo}"
export HF_HOME="${HF_HOME:-/models/hf}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"

SHARED_DIR="$APP_DIR/shared"
if [ ! -d "$SHARED_DIR" ]; then SHARED_DIR="$APP_DIR/../shared"; fi
# Imports overlap downloads; the pipeline waits for the validated snapshot.
exec python -u "$SHARED_DIR/bootstrap.py" --app-dir "$APP_DIR"
