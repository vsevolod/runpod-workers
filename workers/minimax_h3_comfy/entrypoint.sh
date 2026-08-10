#!/bin/bash
# Always emit something to container logs. RunPod UI often shows empty
# "Container" logs when the process exits in the first milliseconds.
set +e
export PYTHONUNBUFFERED=1

boot_log=/tmp/minimax_h3_boot.log
exec > >(tee -a "${boot_log}" /proc/1/fd/1) 2> >(tee -a "${boot_log}" /proc/1/fd/2)

echo "=========================================="
echo "minimax_h3_comfy: ENTRYPOINT $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "minimax_h3_comfy: pwd=$(pwd) uid=$(id -u) PATH=${PATH}"
echo "minimax_h3_comfy: MODEL_DIR=${MODEL_DIR:-<unset>}"
echo "minimax_h3_comfy: which bash=$(command -v bash) python=$(command -v python)"
echo "minimax_h3_comfy: ls /app:"
ls -la /app || true
echo "minimax_h3_comfy: ls /runpod-volume (first 20):"
ls -la /runpod-volume 2>&1 | head -20 || true
echo "minimax_h3_comfy: ls /workspace (first 10):"
ls -la /workspace 2>&1 | head -10 || true
echo "=========================================="

if [[ ! -f /app/start.sh ]]; then
  echo "minimax_h3_comfy: FATAL: /app/start.sh missing"
  sleep 120
  exit 1
fi

# -x traces every command into logs
bash -x /app/start.sh
rc=$?
echo "minimax_h3_comfy: start.sh exited rc=${rc}"
echo "minimax_h3_comfy: ---- boot log tail ----"
tail -n 200 "${boot_log}" 2>/dev/null || true

# Hold process so RunPod scrapes logs before container removal
if [[ "${rc}" -ne 0 ]]; then
  echo "minimax_h3_comfy: sleeping 120s so logs are visible in RunPod UI..."
  sleep 120
fi
exit "${rc}"
