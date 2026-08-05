#!/usr/bin/env python3
"""Download the four MiniMax H3 Comfy-Org weights for T2V v1.

    python download_weights.py --output /runpod-volume/minimax_h3_comfy --dry-run
    python download_weights.py --output /runpod-volume/minimax_h3_comfy

Writes under {output}/models/... only these four files.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HF_REPO = "Comfy-Org/MiniMax-H3"

# (relative path under models/, HF filename)
WEIGHTS: list[tuple[str, str]] = [
    (
        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    ),
    (
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    ),
    (
        "vae/minimax_h3_video_vae_fp16.safetensors",
        "vae/minimax_h3_video_vae_fp16.safetensors",
    ),
    (
        "vae/minimax_h3_audio_vae_fp32.safetensors",
        "vae/minimax_h3_audio_vae_fp32.safetensors",
    ),
]


def expected_targets(output: Path) -> list[Path]:
    models = output / "models"
    return [models / rel for rel, _ in WEIGHTS]


def download_one(repo_id: str, filename: str, dest_dir: Path, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest_dir),
        token=token,
    )
    return Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=os.environ.get("MODEL_DIR", "./models/minimax_h3_comfy"),
        help="Volume root (default: MODEL_DIR or ./models/minimax_h3_comfy)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print target paths only; do not download",
    )
    args = parser.parse_args(argv)

    output = Path(args.output).resolve()
    models_root = output / "models"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_TOKEN")

    targets = expected_targets(output)
    if len(targets) != 4:
        print("internal error: expected exactly four weights", file=sys.stderr)
        return 1

    for t in targets:
        print(t)

    if args.dry_run:
        return 0

    models_root.mkdir(parents=True, exist_ok=True)
    for rel, hf_name in WEIGHTS:
        dest_parent = (models_root / rel).parent
        dest_parent.mkdir(parents=True, exist_ok=True)
        # hf_hub_download with local_dir=models_root keeps repo-relative paths
        path = download_one(HF_REPO, hf_name, models_root, token)
        print(f"ok: {path}")

    # Verify exactly the four expected basenames exist under models/
    missing = [p for p in targets if not p.is_file()]
    if missing:
        print(f"missing after download: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
