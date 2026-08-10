#!/bin/bash
# Create a minimal HF hub cache layout for boot smoke (tiny placeholder files).
# Mirrors real huggingface_hub cache: content lives in blobs/; snapshot entries
# are relative symlinks into ../../blobs/<id> (depth-adjusted for nested paths).
#
# Usage:
#   ./tools/make_fake_hf_cache.sh /tmp/mh3_hub [org/name] [--bytes 64]
#
# Layout:
#   <out>/models--{org}--{name}/refs/main
#   <out>/models--{org}--{name}/blobs/<blob_id>
#   <out>/models--{org}--{name}/snapshots/<hash>/<rel> -> relative symlink to blobs/
set -euo pipefail

OUT="${1:-./fake_hub}"
MODEL_ID="${2:-Comfy-Org/MiniMax-H3}"
BYTES=64
if [[ "${3:-}" == "--bytes" ]]; then
  BYTES="${4:-64}"
elif [[ "${2:-}" == "--bytes" ]]; then
  MODEL_ID="Comfy-Org/MiniMax-H3"
  BYTES="${3:-64}"
fi

if [[ "${MODEL_ID}" != */* ]] || [[ "${MODEL_ID}" == *:* ]]; then
  echo "MODEL_ID must be org/name (got ${MODEL_ID})" >&2
  exit 2
fi

ORG="${MODEL_ID%%/*}"
NAME="${MODEL_ID#*/}"
SNAP_HASH="fakesnap001"
MODEL_ROOT="${OUT}/models--${ORG}--${NAME}"
SNAP="${MODEL_ROOT}/snapshots/${SNAP_HASH}"
BLOBS="${MODEL_ROOT}/blobs"

FILES=(
  "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
  "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
  "vae/minimax_h3_video_vae_fp16.safetensors"
  "vae/minimax_h3_audio_vae_fp32.safetensors"
)

mkdir -p "${MODEL_ROOT}/refs" "${SNAP}" "${BLOBS}"
echo "${SNAP_HASH}" > "${MODEL_ROOT}/refs/main"

i=0
for rel in "${FILES[@]}"; do
  i=$((i + 1))
  # Stable fake blob id (real hub uses content hash; we only need the symlink chain)
  blob_id="$(printf 'fakeblob%02d' "${i}")"
  blob_path="${BLOBS}/${blob_id}"
  dd if=/dev/zero of="${blob_path}" bs=1 count="${BYTES}" status=none 2>/dev/null \
    || head -c "${BYTES}" /dev/zero > "${blob_path}"

  link_path="${SNAP}/${rel}"
  mkdir -p "$(dirname "${link_path}")"
  # Relative path from snapshots/<hash>/<dir...>/file up to model root, then into blobs/
  # depth: snapshots/hash = 2; plus one per directory component of rel
  depth=2
  IFS='/' read -ra parts <<< "${rel}"
  # parts include filename; parent dirs = len-1
  if [[ ${#parts[@]} -gt 1 ]]; then
    depth=$((depth + ${#parts[@]} - 1))
  fi
  up=""
  for ((d = 0; d < depth; d++)); do
    up="${up}../"
  done
  rel_target="${up}blobs/${blob_id}"
  # Replace any existing file/link
  rm -f "${link_path}"
  ln -s "${rel_target}" "${link_path}"
  echo "blob ${blob_path} (${BYTES} bytes) <- ${link_path} -> ${rel_target}"
done

echo "OK fake HF cache root: ${OUT}"
echo "MODEL_NAME=${MODEL_ID}"
echo "Mount as: -v ${OUT}:/runpod-volume/huggingface-cache/hub"
echo "Env:      HF_CACHE_ROOT=/runpod-volume/huggingface-cache/hub MODEL_NAME=${MODEL_ID}"
