#!/usr/bin/env python3
"""Download the official Qwen-Image-2.1-Turbo snapshot before inference starts.

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
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen-Image-2.1-Turbo"

try:
    from qwen_image_21_infer.runtime_config import validate_turbo_snapshot
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from qwen_image_21_infer.runtime_config import validate_turbo_snapshot


def _snapshot_download(model_id: str, local_dir: str, token: str | None) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, local_dir=local_dir, token=token)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=os.environ.get("MODEL_DIR", "./models/qwen_image_21_turbo"),
        help="Diffusers snapshot directory (default: MODEL_DIR or ./models/qwen_image_21_turbo)",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("MODEL_ID", MODEL_ID),
        help=f"Hugging Face repo id (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME", ""),
        help="Optional HF cache root. The snapshot itself is written to --output.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="Hugging Face token (or set HF_TOKEN)",
    )
    args = parser.parse_args(argv)

    out = Path(args.output).expanduser().resolve()
    if (out / "model_index.json").exists():
        try:
            validate_turbo_snapshot(out)
        except ValueError as err:
            print(f"error: {err}. Choose a new --output directory for Turbo.", file=sys.stderr)
            return 1
    out.mkdir(parents=True, exist_ok=True)

    if args.hf_home:
        hf_home = Path(args.hf_home).expanduser().resolve()
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
        print(f"HF_HOME={hf_home}")

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    print(f"MODEL_DIR={out}")
    started = time.monotonic()
    print(f"stage=model_download begin model={args.model_id} path={out}", flush=True)
    try:
        path = _snapshot_download(args.model_id, str(out), args.token)
    except Exception as err:
        print(f"error: snapshot failed: {err}", file=sys.stderr)
        print(
            "If the repo is gated, accept the Qwen Research License on the model "
            "page and pass HF_TOKEN.",
            file=sys.stderr,
        )
        return 1

    try:
        validate_turbo_snapshot(path)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    print(f"  -> {path}")
    print(f"stage=model_download done elapsed_s={time.monotonic() - started:.2f}", flush=True)
    print()
    print("Endpoint env:")
    print(f"  MODEL_DIR={out}")
    print(f"  MODEL_ID={args.model_id}")
    print("  OFFLOAD=cpu")
    print("  LOCAL_FILES_ONLY=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
