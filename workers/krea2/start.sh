#!/bin/sh
set -eu

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export MODEL_DIR="${MODEL_DIR:-/models/krea2}"
export HF_HOME="${HF_HOME:-/models/hf}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"

# Download the selected DiT plus HF text encoder and clean Diffusers VAE.
# Resume incomplete transfers; do not start inference after a failed download.
python -u "$APP_DIR/download_weights.py" --skip-vae-file
export LOCAL_FILES_ONLY=1
exec python -u "$APP_DIR/handler.py"
