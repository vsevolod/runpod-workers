#!/usr/bin/env bash
# Boot pinned ComfyUI then RunPod handler.
set -euo pipefail

MODEL_ROOT="${MODEL_DIR:-/runpod-volume/minimax_h3_comfy}"
COMFYUI_PATH="${COMFYUI_PATH:-/comfyui}"
COMFY_HOST="${COMFY_HOST:-127.0.0.1}"
COMFY_PORT="${COMFY_PORT:-8188}"
export COMFY_URL="${COMFY_URL:-http://${COMFY_HOST}:${COMFY_PORT}}"
export COMFY_OUTPUT_DIR="${COMFY_OUTPUT_DIR:-${COMFYUI_PATH}/output}"

echo "minimax_h3_comfy: MODEL_DIR=${MODEL_ROOT}"
echo "minimax_h3_comfy: COMFYUI_PATH=${COMFYUI_PATH}"

if [[ ! -d "${MODEL_ROOT}/models" ]]; then
  echo "minimax_h3_comfy: missing ${MODEL_ROOT}/models — run download_weights.py first" >&2
  exit 1
fi

# Prefer volume models over image placeholders
if [[ -e "${COMFYUI_PATH}/models" || -L "${COMFYUI_PATH}/models" ]]; then
  rm -rf "${COMFYUI_PATH}/models"
fi
ln -sfn "${MODEL_ROOT}/models" "${COMFYUI_PATH}/models"

mkdir -p "${COMFY_OUTPUT_DIR}"

echo "minimax_h3_comfy: starting ComfyUI on ${COMFY_HOST}:${COMFY_PORT}"
python -u "${COMFYUI_PATH}/main.py" \
  --listen "${COMFY_HOST}" \
  --port "${COMFY_PORT}" \
  --disable-auto-launch \
  --disable-metadata \
  --log-stdout &
echo $! > /tmp/comfyui.pid

echo "minimax_h3_comfy: starting handler"
exec python -u /app/handler.py
