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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --fake-root) FAKE_ROOT="$2"; KEEP_FAKE=1; shift 2 ;;
    -h|--help)
      sed -n '1,20p' "$0"
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

if [[ -z "${FAKE_ROOT}" ]]; then
  FAKE_ROOT="$(mktemp -d /tmp/mh3_fake.XXXXXX)"
fi
log "fake weights -> ${FAKE_ROOT}"
bash "${WORKER_DIR}/tools/make_fake_weights.sh" "${FAKE_ROOT}"

# --- Case 1: missing volume → must fail with clear ERROR + ENTRYPOINT ---
log "=== case: missing_models (expect fail) ==="
set +e
out1="$(docker run --rm \
  -e MODEL_DIR=/runpod-volume/minimax_h3_comfy \
  -e BOOT_FAIL_SLEEP=0 \
  -e BOOT_CHECK=1 \
  "${IMAGE}" 2>&1)"
rc1=$?
set -e
echo "${out1}" | tail -n 30
echo "${out1}" | grep -q "ENTRYPOINT" || die "missing_models: no ENTRYPOINT in logs"
echo "${out1}" | grep -qiE "ERROR:.*models|missing .*models" \
  || die "missing_models: expected models ERROR, got rc=${rc1}"
[[ "${rc1}" -ne 0 ]] || die "missing_models: expected non-zero exit"
log "missing_models OK (rc=${rc1})"

# --- Case 2: fake weights → mock Comfy ready → BOOT_CHECK exit 0 ---
log "=== case: fake_weights_ok (expect success) ==="
set +e
out2="$(docker run --rm \
  -v "${FAKE_ROOT}:/runpod-volume/minimax_h3_comfy:ro" \
  -e MODEL_DIR=/runpod-volume/minimax_h3_comfy \
  -e BOOT_FAIL_SLEEP=0 \
  -e BOOT_CHECK=1 \
  "${IMAGE}" 2>&1)"
rc2=$?
set -e
echo "${out2}" | tail -n 40
echo "${out2}" | grep -q "ENTRYPOINT" || die "fake_weights_ok: no ENTRYPOINT"
echo "${out2}" | grep -q "ok weight:" || die "fake_weights_ok: no ok weight lines"
echo "${out2}" | grep -q "ComfyUI ready" || die "fake_weights_ok: ComfyUI never ready"
echo "${out2}" | grep -q "BOOT_CHECK ok" || die "fake_weights_ok: no BOOT_CHECK ok"
[[ "${rc2}" -eq 0 ]] || die "fake_weights_ok: expected rc=0, got ${rc2}"
log "fake_weights_ok OK"

# --- Case 3: incomplete weights → fail ---
log "=== case: incomplete_weights (expect fail) ==="
BAD="$(mktemp -d /tmp/mh3_bad.XXXXXX)"
mkdir -p "${BAD}/models/diffusion_models"
echo x > "${BAD}/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
set +e
out3="$(docker run --rm \
  -v "${BAD}:/runpod-volume/minimax_h3_comfy:ro" \
  -e MODEL_DIR=/runpod-volume/minimax_h3_comfy \
  -e BOOT_FAIL_SLEEP=0 \
  -e BOOT_CHECK=1 \
  "${IMAGE}" 2>&1)"
rc3=$?
set -e
echo "${out3}" | tail -n 25
echo "${out3}" | grep -q "MISSING weight" || die "incomplete_weights: expected MISSING weight"
[[ "${rc3}" -ne 0 ]] || die "incomplete_weights: expected failure"
rm -rf "${BAD}"
log "incomplete_weights OK"

if [[ "${KEEP_FAKE}" -eq 0 ]]; then
  rm -rf "${FAKE_ROOT}"
fi

log "ALL BOOT SMOKE CHECKS PASSED"
log "Next (optional, heavy): full CUDA image with real weights on a big GPU"
echo "  docker build -f workers/minimax_h3_comfy/Dockerfile -t minimax-h3-comfy:local ."
