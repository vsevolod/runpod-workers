#!/usr/bin/env python3
"""Bootstrap Krea 2 FP8 + TE/VAE caches onto a Network Volume (or local dir).

Run on a Pod with the volume mounted, or any machine with disk + HF access:

    python download_weights.py --output /runpod-volume/krea2

Does NOT bake weights into the Docker image (prefer volume for ~18 GB DiT/VAE
plus HF text-encoder cache).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DIT_REPO = "AlperKTS/Krea2_FP8"
DIT_FILE = "krea2_turbo_fp8.safetensors"
VAE_FILE = "qwen_image_vae.safetensors"
TEXT_ENCODER_ID = "Qwen/Qwen3-VL-4B-Instruct"
VAE_REPO = "Qwen/Qwen-Image"


def _hf_download(repo_id: str, filename: str, dest_dir: Path, token: str | None) -> Path:
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
        default=os.environ.get("MODEL_DIR", "./models/krea2"),
        help="Directory for DiT/VAE safetensors (default: MODEL_DIR or ./models/krea2)",
    )
    parser.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME", ""),
        help="Optional HF cache root (text encoder + VAE config). "
        "If empty, uses default huggingface cache.",
    )
    parser.add_argument(
        "--skip-dit",
        action="store_true",
        help="Skip AlperKTS DiT download",
    )
    parser.add_argument(
        "--skip-vae-file",
        action="store_true",
        help="Skip standalone VAE safetensors (still pulls HF VAE for offline use)",
    )
    parser.add_argument(
        "--skip-text-encoder",
        action="store_true",
        help="Skip Qwen3-VL-4B-Instruct snapshot",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="Hugging Face token (or set HF_TOKEN)",
    )
    args = parser.parse_args(argv)

    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.hf_home:
        hf_home = Path(args.hf_home).expanduser().resolve()
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
        print(f"HF_HOME={hf_home}")

    print(f"MODEL_DIR={out}")

    if not args.skip_dit:
        print(f"Downloading DiT {DIT_REPO}/{DIT_FILE} …")
        dit = _hf_download(DIT_REPO, DIT_FILE, out, args.token)
        print(f"  -> {dit}")

    if not args.skip_vae_file:
        # Prefer community single-file if present on the same repo; otherwise HF repo.
        try:
            print(f"Trying VAE {DIT_REPO}/{VAE_FILE} …")
            vae = _hf_download(DIT_REPO, VAE_FILE, out, args.token)
            print(f"  -> {vae}")
        except Exception as err:
            print(f"  AlperKTS VAE not available ({err}); will use {VAE_REPO} cache only")

    if not args.skip_text_encoder:
        from huggingface_hub import snapshot_download

        print(f"Snapshot {TEXT_ENCODER_ID} …")
        te_path = snapshot_download(TEXT_ENCODER_ID, token=args.token)
        print(f"  -> {te_path}")

        print(f"Snapshot {VAE_REPO} (vae subfolder) …")
        vae_repo_path = snapshot_download(
            VAE_REPO,
            allow_patterns=["vae/*", "VAE/*", "*.json"],
            token=args.token,
        )
        print(f"  -> {vae_repo_path}")

    print("Done.")
    print()
    print("Volume layout (recommended):")
    print(f"  {out}/{DIT_FILE}")
    print(f"  {out}/{VAE_FILE}   # optional if HF VAE cache is warm")
    print("Env on endpoint:")
    print(f"  MODEL_DIR={out}")
    if args.hf_home:
        print(f"  HF_HOME={args.hf_home}")
        print("  LOCAL_FILES_ONLY=1   # after first successful download")
    return 0


if __name__ == "__main__":
    sys.exit(main())
