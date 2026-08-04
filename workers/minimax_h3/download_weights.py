#!/usr/bin/env python3
"""Bootstrap MiniMax-H3 t2va weights onto a Network Volume (or local dir).

Run on a Pod with the volume mounted (not inside a serverless job):

    python download_weights.py --output /runpod-volume/minimax_h3

Does NOT bake weights into the Docker image (~144 GB t2va half).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "MiniMaxAI/MiniMax-H3"

# snapshot_download allow_patterns — t2va half only (converted layout).
# Ref2VA / transformer_ref / original FL2VA trees are NOT matched → not downloaded.
ALLOW_PATTERNS_T2VA = [
    "modular_model_index.json",
    "model_index.json",
    "transformer/*",
    "transformer/**",
    "text_encoder/*",
    "text_encoder/**",
    "vae/*",
    "vae/**",
    "audio_vae/*",
    "audio_vae/**",
    "tokenizer/*",
    "tokenizer/**",
    "processor/*",
    "processor/**",
    "scheduler/*",
    "scheduler/**",
    "audio_scheduler/*",
    "audio_scheduler/**",
]


def _snapshot_download(*args, **kwargs):
    """Indirection so unit tests can mock without importing huggingface_hub."""
    from huggingface_hub import snapshot_download

    return snapshot_download(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=os.environ.get("MODEL_DIR", "./models/minimax_h3"),
        help="Directory for HF snapshot (default: MODEL_DIR or ./models/minimax_h3)",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("MODEL_ID", REPO_ID),
        help=f"HF repo id (default: {REPO_ID})",
    )
    parser.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME", ""),
        help="Optional HF cache root. If empty, uses default huggingface cache.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="Hugging Face token (or set HF_TOKEN)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned download and exit without network",
    )
    args = parser.parse_args(argv)

    out = Path(args.output).expanduser().resolve()

    if args.hf_home:
        hf_home = Path(args.hf_home).expanduser().resolve()
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
        print(f"HF_HOME={hf_home}")

    print(f"Repo: {args.repo}")
    print(f"Output: {out}")
    print(f"Allow patterns ({len(ALLOW_PATTERNS_T2VA)}):")
    for p in ALLOW_PATTERNS_T2VA:
        print(f"  - {p}")

    if args.dry_run:
        print("Dry-run: not calling snapshot_download.")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    # Optional high-performance HF xet path (not hf_transfer).
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    print(f"Snapshot {args.repo} -> {out} …")
    path = _snapshot_download(
        args.repo,
        local_dir=str(out),
        token=args.token,
        allow_patterns=list(ALLOW_PATTERNS_T2VA),
    )
    print(f"  -> {path}")

    print("Done.")
    print()
    print("Volume layout (t2va half):")
    print(f"  {out}/modular_model_index.json")
    print(f"  {out}/transformer/ …")
    print(f"  {out}/text_encoder/ …")
    print("Env on endpoint:")
    print(f"  MODEL_DIR={out}")
    print("  LOCAL_FILES_ONLY=1   # after first successful download")
    print("  # all four BUCKET_* for MP4 URL delivery")
    return 0


if __name__ == "__main__":
    sys.exit(main())
