#!/usr/bin/env python3
"""Download Qwen-Image-2.1-Turbo and the selected VAE before inference starts.

The container entrypoint runs this on every fresh worker. For a manual download:

    pip install huggingface-hub hf_transfer
    export HF_TOKEN=hf_...   # accept the model license on Hugging Face first
    python download_weights.py --output /models/qwen_image_21_turbo

Does not bake weights into the Docker image.
The checkpoint contains the accelerated weights and eight-step sampling schedule.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

workers_dir = Path(__file__).resolve().parents[1]
if (workers_dir / "shared").is_dir():
    sys.path.insert(0, str(workers_dir))
from shared.model_download import ModelDownloader, add_download_arguments
from shared.prefetch import ComponentPublisher

MODEL_ID = "Qwen/Qwen-Image-2.1-Turbo"

try:
    from qwen_image_21_infer.runtime_config import (
        TEXTURE_FIX_VAE_DIR, TEXTURE_FIX_VAE_ID, resolve_vae_variant, validate_turbo_snapshot,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from qwen_image_21_infer.runtime_config import (
        TEXTURE_FIX_VAE_DIR, TEXTURE_FIX_VAE_ID, resolve_vae_variant, validate_turbo_snapshot,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_download_arguments(parser, default_output="./models/qwen_image_21_turbo")
    parser.add_argument(
        "--model-id",
        default=os.environ.get("MODEL_ID", MODEL_ID),
        help=f"Hugging Face repo id (default: {MODEL_ID})",
    )
    args = parser.parse_args(argv)

    try:
        vae_variant = resolve_vae_variant()
    except ValueError as err:
        parser.error(str(err))

    out = Path(args.output).expanduser().resolve()
    if (out / "model_index.json").exists():
        try:
            validate_turbo_snapshot(out)
        except ValueError as err:
            print(f"error: {err}. Choose a new --output directory for Turbo.", file=sys.stderr)
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
            path = downloads.snapshot(args.model_id, local_dir=out, allow_patterns=patterns)
            validate_turbo_snapshot(path)
            if vae_variant == "texture-fix":
                downloads.snapshot(
                    TEXTURE_FIX_VAE_ID, local_dir=out / TEXTURE_FIX_VAE_DIR,
                    allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"],
                )
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
