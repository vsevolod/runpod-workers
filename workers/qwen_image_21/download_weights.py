#!/usr/bin/env python3
"""Download only the request-selected Qwen-Image-2.1 variant and its VAE.

The handler starts this when local files are missing/incomplete. For a manual download:

    pip install huggingface-hub hf_transfer
    export HF_TOKEN=hf_...   # accept the model license on Hugging Face first
    python download_weights.py --output /models/qwen_image_21_turbo

Does not bake weights into the Docker image.
Turbo owns an eight-step schedule; Base uses the worker's 40-step Euler/simple grid.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
workers_dir = Path(__file__).resolve().parents[1]
if (workers_dir / "shared").is_dir():
    sys.path.insert(0, str(workers_dir))
from shared.model_download import ModelDownloader, add_download_arguments
from shared.prefetch import ComponentPublisher
from qwen_image_21_infer.model_store import snapshot_ready, write_manifest, remove_incomplete_files

MODEL_ID = "Qwen/Qwen-Image-2.1-Turbo"

from qwen_image_21_infer.runtime_config import (
    TEXTURE_FIX_VAE_DIR, TEXTURE_FIX_VAE_ID, resolve_vae_variant, validate_model_snapshot,
    resolve_model_id,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_download_arguments(parser, default_output="./models/qwen_image_21_turbo")
    parser.add_argument("--model-variant", choices=("turbo", "base"), default="turbo")
    parser.add_argument("--vae-dir", default=None, help="Shared local texture-fix VAE directory")
    parser.add_argument(
        "--model-id",
        default=None,
        help=f"Hugging Face repo id (default: {MODEL_ID})",
    )
    args = parser.parse_args(argv)
    args.model_id = args.model_id or resolve_model_id(args.model_variant)

    try:
        vae_variant = resolve_vae_variant()
    except ValueError as err:
        parser.error(str(err))

    out = Path(args.output).expanduser().resolve()
    if (out / "model_index.json").exists():
        try:
            validate_model_snapshot(out, args.model_variant)
        except ValueError as err:
            print(f"error: {err}. Choose a separate --output directory for {args.model_variant}.", file=sys.stderr)
            return 1
    print(f"MODEL_DIR={out}")
    publisher = ComponentPublisher(args.prefetch_state_dir)
    try:
        with ModelDownloader(out, hf_home=args.hf_home, token=args.token,
                             pin_revisions=bool(args.prefetch_state_dir)) as downloads:
            # Supplying the alternative VAE to from_pretrained avoids loading
            # the stock VAE. Do not download those unused weights either.
            patterns = None if vae_variant == "original" else [
                "model_index.json", "scheduler/*", "processor/*",
                "text_encoder/*", "transformer/*",
            ]
            if not snapshot_ready(out, model_id=args.model_id, model_variant=args.model_variant, profile=vae_variant):
                remove_incomplete_files(out)
                path = downloads.snapshot(args.model_id, local_dir=out, allow_patterns=patterns)
                validate_model_snapshot(path, args.model_variant)
                write_manifest(path, model_id=args.model_id, model_variant=args.model_variant,
                               profile=vae_variant, revision=downloads.revisions.get(args.model_id))
            if vae_variant == "texture-fix":
                vae_dir = Path(args.vae_dir) if args.vae_dir else out / TEXTURE_FIX_VAE_DIR
                if not snapshot_ready(vae_dir, model_id=TEXTURE_FIX_VAE_ID, model_variant="vae"):
                    remove_incomplete_files(vae_dir)
                    downloads.snapshot(
                        TEXTURE_FIX_VAE_ID, local_dir=vae_dir,
                        allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"],
                    )
                    write_manifest(vae_dir, model_id=TEXTURE_FIX_VAE_ID, model_variant="vae",
                                   revision=downloads.revisions.get(TEXTURE_FIX_VAE_ID))
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"error: snapshot failed: {err}", file=sys.stderr)
        print(
            "If the repo is gated, accept the Qwen Research License on the model "
            "page and pass HF_TOKEN.",
            file=sys.stderr,
        )
        return 1

    publisher.ready("all", out)
    print()
    print("Endpoint env:")
    print(f"  MODEL_DIR={out}")
    print(f"  MODEL_ID={args.model_id}")
    print(f"  QWEN_VAE={vae_variant}")
    print("  OFFLOAD=cpu")
    print("  LOCAL_FILES_ONLY=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
