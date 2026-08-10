#!/bin/bash
# Create the four expected weight paths (tiny placeholder files).
# Usage:
#   ./tools/make_fake_weights.sh /tmp/mh3_fake
#   ./tools/make_fake_weights.sh /tmp/mh3_fake --bytes 1024
set -euo pipefail

OUT="${1:-./fake_volume/minimax_h3_comfy}"
BYTES="${2:-64}"
if [[ "${BYTES}" == "--bytes" ]]; then
  BYTES="${3:-64}"
fi

FILES=(
  "models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
  "models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
  "models/vae/minimax_h3_video_vae_fp16.safetensors"
  "models/vae/minimax_h3_audio_vae_fp32.safetensors"
)

mkdir -p "${OUT}"
for rel in "${FILES[@]}"; do
  path="${OUT}/${rel}"
  mkdir -p "$(dirname "${path}")"
  # small non-empty file (start.sh checks -f and du)
  dd if=/dev/zero of="${path}" bs=1 count="${BYTES}" status=none 2>/dev/null \
    || head -c "${BYTES}" /dev/zero > "${path}"
  echo "wrote ${path} (${BYTES} bytes)"
done

echo "OK fake volume root: ${OUT}"
echo "Mount as: -v ${OUT}:/runpod-volume/minimax_h3_comfy"
echo "Env:      MODEL_DIR=/runpod-volume/minimax_h3_comfy"
