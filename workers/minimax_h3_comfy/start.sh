#!/bin/bash
# Boot pinned ComfyUI then RunPod handler.
# Fail loud with diagnostics — silent exit marks workers "unhealthy".
set -euo pipefail

# Plain stdout — reliable under Docker log capture (no /dev/stderr tee).
log() { echo "minimax_h3_comfy: $*"; }
die() { echo "minimax_h3_comfy: ERROR: $*"; exit 1; }

COMFYUI_PATH="${COMFYUI_PATH:-/comfyui}"
COMFY_HOST="${COMFY_HOST:-127.0.0.1}"
COMFY_PORT="${COMFY_PORT:-8188}"
export COMFY_URL="${COMFY_URL:-http://${COMFY_HOST}:${COMFY_PORT}}"
export COMFY_OUTPUT_DIR="${COMFY_OUTPUT_DIR:-${COMFYUI_PATH}/output}"
# Hard-coded materialize root (model_store.py contract; not configurable in v1)
MODELS_ROOT="/models"

log "boot pid=$$"
log "MODEL_NAME=${MODEL_NAME:-}"
log "HF_CACHE_ROOT=${HF_CACHE_ROOT:-/runpod-volume/huggingface-cache/hub}"
log "MODEL_DIR=${MODEL_DIR:-}"
log "COMFYUI_PATH=${COMFYUI_PATH}"
log "COMFY_URL=${COMFY_URL}"
log "python=$(command -v python || true) $(python -V 2>&1 || true)"
log "gcc=$(command -v gcc || echo MISSING)"

# --- volume / cache diagnostics ------------------------------------------
log "mount hints:"
ls -la /runpod-volume 2>/dev/null | head -20 || log "  /runpod-volume not present"
ls -la /workspace 2>/dev/null | head -10 || log "  /workspace not present"
if [[ -d /runpod-volume/huggingface-cache ]]; then
  log "huggingface-cache present:"
  ls -la /runpod-volume/huggingface-cache 2>/dev/null | head -10 || true
fi

# Materialize four weights into /models via symlinks (cache preferred, legacy volume fallback).
# Shell does not parse JSON; Python always writes /models or exits non-zero.
log "running model_store.py"
python -u /app/model_store.py || die "model_store.py failed — set endpoint Model/MODEL_NAME for HF cache, or attach Network Volume + MODEL_DIR (no runtime download in v1)"

REQUIRED=(
  "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
  "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
  "vae/minimax_h3_video_vae_fp16.safetensors"
  "vae/minimax_h3_audio_vae_fp32.safetensors"
)
missing=0
for rel in "${REQUIRED[@]}"; do
  if [[ ! -f "${MODELS_ROOT}/${rel}" ]]; then
    log "MISSING weight: ${MODELS_ROOT}/${rel}"
    missing=1
  else
    # -L follows symlinks so sizes reflect real targets (V2.3)
    log "ok weight: ${rel} ($(du -hL "${MODELS_ROOT}/${rel}" 2>/dev/null | awk '{print $1}'))"
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  die "one or more required weights missing under ${MODELS_ROOT}"
fi

# /comfyui/models is a real directory in the image; ln -sfn cannot replace a directory.
if [[ -e "${COMFYUI_PATH}/models" || -L "${COMFYUI_PATH}/models" ]]; then
  rm -rf "${COMFYUI_PATH}/models"
fi
ln -sfn "${MODELS_ROOT}" "${COMFYUI_PATH}/models"
log "symlinked ${COMFYUI_PATH}/models -> ${MODELS_ROOT}"
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
  # Production Comfy image has a real models/ directory; we must have replaced it.
  if [[ ! -L "${COMFYUI_PATH}/models" ]]; then
    die "BOOT_CHECK: ${COMFYUI_PATH}/models is not a symlink (rm -rf + ln -sfn failed?)"
  fi
  models_link="$(readlink "${COMFYUI_PATH}/models")"
  if [[ "${models_link}" != "${MODELS_ROOT}" && "${models_link}" != "${MODELS_ROOT}/" ]]; then
    die "BOOT_CHECK: ${COMFYUI_PATH}/models -> ${models_link}, expected ${MODELS_ROOT}"
  fi
  # Sentinel only exists in Dockerfile.bootcheck real models dir; must be gone after replace.
  if [[ -e "${COMFYUI_PATH}/models/.bootcheck_sentinel" ]]; then
    die "BOOT_CHECK: .bootcheck_sentinel still visible under ${COMFYUI_PATH}/models (directory not replaced)"
  fi
  log "comfy models link OK: ${COMFYUI_PATH}/models -> ${models_link} (stock dir replaced)"
  log "BOOT_CHECK ok — models present, Comfy ready; skipping runpod.serverless.start"
  # Do not leave tail -F on stdout — it keeps entrypoint pipes open forever.
  exit 0
fi

# Stream Comfy logs to a file only (never stdout) so Docker/entrypoint pipes close.
tail -n +1 -F /tmp/comfyui.log >>/tmp/comfyui_follow.log 2>&1 &

log "starting handler"
exec python -u /app/handler.py
