#!/bin/bash
# Fast local validation of container boot path (no RunPod, no CUDA, no HF).
#
# From monorepo root:
#   ./workers/minimax_h3_comfy/tools/local_boot_smoke.sh
# From this worker dir:
#   ./tools/local_boot_smoke.sh
set -euo pipefail

WORKER_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${BOOTCHECK_IMAGE:-minimax-h3-bootcheck:local}"
FAKE_ROOT="${FAKE_ROOT:-}"
KEEP_FAKE=0
MODEL_ID="Comfy-Org/MiniMax-H3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --fake-root) FAKE_ROOT="$2"; KEEP_FAKE=1; shift 2 ;;
    -h|--help)
      sed -n '1,25p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

log() { echo "[boot_smoke] $*"; }
die() { echo "[boot_smoke] FAIL: $*" >&2; exit 1; }

log "worker dir: ${WORKER_DIR}"
log "building ${IMAGE} (slim, ~seconds after first pull)..."
docker build -f "${WORKER_DIR}/Dockerfile.bootcheck" -t "${IMAGE}" "${WORKER_DIR}"

TMP_BASE="$(mktemp -d /tmp/mh3_boot.XXXXXX)"
cleanup() {
  if [[ "${KEEP_FAKE}" -eq 0 ]]; then
    rm -rf "${TMP_BASE}"
  fi
}
trap cleanup EXIT

# --- Case 1: no cache, no volume → must fail with clear ERROR + ENTRYPOINT ---
log "=== case: missing_source (expect fail) ==="
set +e
out1="$(docker run --rm \
  -e MODEL_NAME="${MODEL_ID}" \
  -e HF_CACHE_ROOT=/runpod-volume/huggingface-cache/hub \
  -e BOOT_FAIL_SLEEP=0 \
  -e BOOT_CHECK=1 \
  "${IMAGE}" 2>&1)"
rc1=$?
set -e
echo "${out1}" | tail -n 30
echo "${out1}" | grep -q "ENTRYPOINT" || die "missing_source: no ENTRYPOINT in logs"
echo "${out1}" | grep -qiE "ERROR|No usable model source|model_store" \
  || die "missing_source: expected model_store ERROR, got rc=${rc1}"
[[ "${rc1}" -ne 0 ]] || die "missing_source: expected non-zero exit"
log "missing_source OK (rc=${rc1})"

# --- Case 2: fake HF cache → source=cache → mock Comfy ready ---
log "=== case: fake_cache_ok (expect success, source=cache) ==="
HUB="${TMP_BASE}/hub"
bash "${WORKER_DIR}/tools/make_fake_hf_cache.sh" "${HUB}" "${MODEL_ID}"
set +e
out2="$(docker run --rm \
  -v "${HUB}:/runpod-volume/huggingface-cache/hub:ro" \
  -e MODEL_NAME="${MODEL_ID}" \
  -e HF_CACHE_ROOT=/runpod-volume/huggingface-cache/hub \
  -e BOOT_FAIL_SLEEP=0 \
  -e BOOT_CHECK=1 \
  "${IMAGE}" 2>&1)"
rc2=$?
set -e
echo "${out2}" | tail -n 40
echo "${out2}" | grep -q "ENTRYPOINT" || die "fake_cache_ok: no ENTRYPOINT"
echo "${out2}" | grep -q "source=cache" || die "fake_cache_ok: expected [ModelStore] source=cache"
echo "${out2}" | grep -q "ok weight:" || die "fake_cache_ok: no ok weight lines"
echo "${out2}" | grep -q "ComfyUI ready" || die "fake_cache_ok: ComfyUI never ready"
echo "${out2}" | grep -q "comfy models link OK" || die "fake_cache_ok: stock /comfyui/models was not replaced with symlink"
echo "${out2}" | grep -q "BOOT_CHECK ok" || die "fake_cache_ok: no BOOT_CHECK ok"
# Fixture must use snapshot → blobs relative symlinks (not plain files in snapshot/)
[[ -d "${HUB}/models--Comfy-Org--MiniMax-H3/blobs" ]] \
  || die "fake_cache_ok: expected blobs/ in fake HF cache"
snap_file="${HUB}/models--Comfy-Org--MiniMax-H3/snapshots/fakesnap001/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
[[ -L "${snap_file}" ]] || die "fake_cache_ok: snapshot entry must be symlink into blobs/"
[[ "${rc2}" -eq 0 ]] || die "fake_cache_ok: expected rc=0, got ${rc2}"
log "fake_cache_ok OK"

# --- Case 3: no cache, fake legacy volume → source=volume ---
log "=== case: fake_volume_ok (expect success, source=volume) ==="
if [[ -z "${FAKE_ROOT}" ]]; then
  FAKE_ROOT="${TMP_BASE}/volume"
fi
bash "${WORKER_DIR}/tools/make_fake_weights.sh" "${FAKE_ROOT}"
set +e
out3="$(docker run --rm \
  -v "${FAKE_ROOT}:/runpod-volume/minimax_h3_comfy:ro" \
  -e MODEL_NAME="${MODEL_ID}" \
  -e HF_CACHE_ROOT=/runpod-volume/huggingface-cache/hub \
  -e MODEL_DIR=/runpod-volume/minimax_h3_comfy \
  -e BOOT_FAIL_SLEEP=0 \
  -e BOOT_CHECK=1 \
  "${IMAGE}" 2>&1)"
rc3=$?
set -e
echo "${out3}" | tail -n 40
echo "${out3}" | grep -q "ENTRYPOINT" || die "fake_volume_ok: no ENTRYPOINT"
echo "${out3}" | grep -q "source=volume" || die "fake_volume_ok: expected [ModelStore] source=volume"
echo "${out3}" | grep -q "ok weight:" || die "fake_volume_ok: no ok weight lines"
echo "${out3}" | grep -q "ComfyUI ready" || die "fake_volume_ok: ComfyUI never ready"
echo "${out3}" | grep -q "comfy models link OK" || die "fake_volume_ok: stock /comfyui/models was not replaced with symlink"
echo "${out3}" | grep -q "BOOT_CHECK ok" || die "fake_volume_ok: no BOOT_CHECK ok"
[[ "${rc3}" -eq 0 ]] || die "fake_volume_ok: expected rc=0, got ${rc3}"
log "fake_volume_ok OK"

# --- Case 4: incomplete volume weights → fail ---
log "=== case: incomplete_weights (expect fail) ==="
BAD="${TMP_BASE}/bad_vol"
mkdir -p "${BAD}/models/diffusion_models"
echo x > "${BAD}/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
set +e
out4="$(docker run --rm \
  -v "${BAD}:/runpod-volume/minimax_h3_comfy:ro" \
  -e MODEL_NAME="${MODEL_ID}" \
  -e HF_CACHE_ROOT=/runpod-volume/huggingface-cache/hub \
  -e MODEL_DIR=/runpod-volume/minimax_h3_comfy \
  -e BOOT_FAIL_SLEEP=0 \
  -e BOOT_CHECK=1 \
  "${IMAGE}" 2>&1)"
rc4=$?
set -e
echo "${out4}" | tail -n 25
echo "${out4}" | grep -qiE "MISSING weight|incomplete|No usable model source|ERROR" \
  || die "incomplete_weights: expected failure signature"
[[ "${rc4}" -ne 0 ]] || die "incomplete_weights: expected failure"
log "incomplete_weights OK"

log "ALL BOOT SMOKE CHECKS PASSED"
log "Next (optional, heavy): full CUDA image with real weights on a big GPU"
echo "  docker build -f workers/minimax_h3_comfy/Dockerfile -t minimax-h3-comfy:local ."
