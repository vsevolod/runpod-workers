#!/bin/bash
# Always emit something to container logs.
# Do not leave background jobs attached to stdout (hangs docker log pipes).
set +e
export PYTHONUNBUFFERED=1

boot_log=/tmp/minimax_h3_boot.log
: >"${boot_log}"

emit() {
  echo "$*" | tee -a "${boot_log}"
}

emit "=========================================="
emit "minimax_h3_comfy: ENTRYPOINT $(date -u +%Y-%m-%dT%H:%M:%SZ)"
emit "minimax_h3_comfy: pwd=$(pwd) uid=$(id -u)"
emit "minimax_h3_comfy: MODEL_DIR=${MODEL_DIR:-<unset>}"
emit "minimax_h3_comfy: BOOT_CHECK=${BOOT_CHECK:-0} BOOT_FAIL_SLEEP=${BOOT_FAIL_SLEEP:-120}"
emit "minimax_h3_comfy: which bash=$(command -v bash) python=$(command -v python)"
emit "minimax_h3_comfy: ls /app:"
ls -la /app 2>&1 | tee -a "${boot_log}" || true
emit "minimax_h3_comfy: ls /runpod-volume (first 20):"
ls -la /runpod-volume 2>&1 | head -20 | tee -a "${boot_log}" || true
emit "minimax_h3_comfy: ls /workspace (first 10):"
ls -la /workspace 2>&1 | head -10 | tee -a "${boot_log}" || true
emit "=========================================="

if [[ ! -f /app/start.sh ]]; then
  emit "minimax_h3_comfy: FATAL: /app/start.sh missing"
  sleep "${BOOT_FAIL_SLEEP:-120}"
  exit 1
fi

# Run start.sh with its own log file (not a live pipe held by children).
set -o pipefail
bash -x /app/start.sh >"${boot_log}.run" 2>&1
rc=$?
set +o pipefail
cat "${boot_log}.run" | tee -a "${boot_log}"

emit "minimax_h3_comfy: start.sh exited rc=${rc}"

if [[ "${rc}" -ne 0 ]]; then
  fail_sleep="${BOOT_FAIL_SLEEP:-120}"
  emit "minimax_h3_comfy: sleeping ${fail_sleep}s so logs are visible in RunPod UI..."
  sleep "${fail_sleep}"
fi
exit "${rc}"
