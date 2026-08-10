#!/bin/bash
# Boot pinned ComfyUI then RunPod handler.
# Fail loud with diagnostics — silent exit marks workers "unhealthy".
set -euo pipefail

# Plain stdout — reliable under Docker log capture (no /dev/stderr tee).
log() { echo "minimax_h3_comfy: $*"; }
die() { echo "minimax_h3_comfy: ERROR: $*"; exit 1; }

MODEL_ROOT="${MODEL_DIR:-/runpod-volume/minimax_h3_comfy}"
COMFYUI_PATH="${COMFYUI_PATH:-/comfyui}"
COMFY_HOST="${COMFY_HOST:-127.0.0.1}"
COMFY_PORT="${COMFY_PORT:-8188}"
export COMFY_URL="${COMFY_URL:-http://${COMFY_HOST}:${COMFY_PORT}}"
export COMFY_OUTPUT_DIR="${COMFY_OUTPUT_DIR:-${COMFYUI_PATH}/output}"

log "boot pid=$$"
log "MODEL_DIR=${MODEL_ROOT}"
log "COMFYUI_PATH=${COMFYUI_PATH}"
log "COMFY_URL=${COMFY_URL}"
log "python=$(command -v python || true) $(python -V 2>&1 || true)"
log "gcc=$(command -v gcc || echo MISSING)"

# --- volume / weights diagnostics ------------------------------------------
log "mount hints:"
ls -la /runpod-volume 2>/dev/null | head -20 || log "  /runpod-volume not present"
ls -la /workspace 2>/dev/null | head -10 || log "  /workspace not present"

# If MODEL_DIR is wrong but weights exist on common volume roots, adopt them.
if [[ ! -d "${MODEL_ROOT}/models" ]]; then
  for candidate in \
    /runpod-volume/minimax_h3_comfy \
    /workspace/minimax_h3_comfy \
    /runpod-volume \
    /workspace
  do
    if [[ -d "${candidate}/models/diffusion_models" ]] || [[ -d "${candidate}/models" ]]; then
      log "MODEL_DIR missing models/; found candidate ${candidate}"
      MODEL_ROOT="${candidate}"
      export MODEL_DIR="${MODEL_ROOT}"
      break
    fi
  done
fi

if [[ ! -d "${MODEL_ROOT}/models" ]]; then
  log "listing MODEL_ROOT parent for debug:"
  ls -la "$(dirname "${MODEL_ROOT}")" 2>/dev/null || true
  die "missing ${MODEL_ROOT}/models — attach Network Volume and run download_weights.py. Set MODEL_DIR to the volume root that contains models/ (e.g. /runpod-volume/minimax_h3_comfy or /workspace/minimax_h3_comfy)"
fi

REQUIRED=(
  "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
  "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
  "vae/minimax_h3_video_vae_fp16.safetensors"
  "vae/minimax_h3_audio_vae_fp32.safetensors"
)
missing=0
for rel in "${REQUIRED[@]}"; do
  if [[ ! -f "${MODEL_ROOT}/models/${rel}" ]]; then
    log "MISSING weight: ${MODEL_ROOT}/models/${rel}"
    missing=1
  else
    log "ok weight: ${rel} ($(du -h "${MODEL_ROOT}/models/${rel}" | awk '{print $1}'))"
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  die "one or more required weights missing under ${MODEL_ROOT}/models"
fi

# Prefer volume models over image placeholders
if [[ -e "${COMFYUI_PATH}/models" || -L "${COMFYUI_PATH}/models" ]]; then
  rm -rf "${COMFYUI_PATH}/models"
fi
ln -sfn "${MODEL_ROOT}/models" "${COMFYUI_PATH}/models"
log "symlinked ${COMFYUI_PATH}/models -> ${MODEL_ROOT}/models"
ls -la "${COMFYUI_PATH}/models" | head -10

mkdir -p "${COMFY_OUTPUT_DIR}"

# --- ComfyUI ---------------------------------------------------------------
if [[ ! -f "${COMFYUI_PATH}/main.py" ]]; then
  die "ComfyUI main.py not found at ${COMFYUI_PATH}/main.py (image build broken)"
fi

log "starting ComfyUI on ${COMFY_HOST}:${COMFY_PORT}"
python -u "${COMFYUI_PATH}/main.py" \
  --listen "${COMFY_HOST}" \
  --port "${COMFY_PORT}" \
  --disable-auto-launch \
  --disable-metadata \
  --log-stdout \
  > /tmp/comfyui.log 2>&1 &
COMFY_PID=$!
echo "${COMFY_PID}" > /tmp/comfyui.pid
log "ComfyUI pid=${COMFY_PID}"

# Brief readiness probe so we fail before RunPod marks "unhealthy" with no logs
ready=0
for i in $(seq 1 120); do
  if ! kill -0 "${COMFY_PID}" 2>/dev/null; then
    log "ComfyUI exited early; last log lines:"
    tail -n 80 /tmp/comfyui.log || true
    die "ComfyUI process died (pid ${COMFY_PID})"
  fi
  if python -c "import urllib.request; urllib.request.urlopen('${COMFY_URL}/system_stats', timeout=2).read()" 2>/dev/null; then
    log "ComfyUI ready after ${i}s"
    ready=1
    break
  fi
  sleep 1
done
if [[ "${ready}" -ne 1 ]]; then
  log "ComfyUI not ready in 120s; last log lines:"
  tail -n 80 /tmp/comfyui.log || true
  die "ComfyUI failed to become ready"
fi

# Local/CI boot smoke: validate weights + Comfy ready without RunPod handler.
if [[ "${BOOT_CHECK:-0}" == "1" ]]; then
  log "BOOT_CHECK ok — models present, Comfy ready; skipping runpod.serverless.start"
  # Do not leave tail -F on stdout — it keeps entrypoint pipes open forever.
  exit 0
fi

# Stream Comfy logs to a file only (never stdout) so Docker/entrypoint pipes close.
tail -n +1 -F /tmp/comfyui.log >>/tmp/comfyui_follow.log 2>&1 &

log "starting handler"
exec python -u /app/handler.py
