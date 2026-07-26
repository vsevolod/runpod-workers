#!/usr/bin/env python3
"""Bootstrap JoyCaption weights onto a Network Volume (or local dir).

Run on a Pod with the volume mounted (not inside a serverless job):

    python download_weights.py --output /runpod-volume/joycaption

Does NOT bake weights into the Docker image (~16 GB snapshot).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_MODEL_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=os.environ.get("MODEL_DIR", "./models/joycaption"),
        help="Directory for HF snapshot (default: MODEL_DIR or ./models/joycaption)",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("MODEL_ID", DEFAULT_MODEL_ID),
        help=f"HF model id (default: {DEFAULT_MODEL_ID})",
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
    args = parser.parse_args(argv)

    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.hf_home:
        hf_home = Path(args.hf_home).expanduser().resolve()
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
        print(f"HF_HOME={hf_home}")

    from huggingface_hub import snapshot_download

    print(f"Snapshot {args.model_id} -> {out} …")
    path = snapshot_download(
        args.model_id,
        local_dir=str(out),
        token=args.token,
    )
    print(f"  -> {path}")

    print("Done.")
    print()
    print("Volume layout:")
    print(f"  {out}/config.json")
    print(f"  {out}/model-00001-of-00004.safetensors …")
    print("Env on endpoint:")
    print(f"  MODEL_DIR={out}")
    print(f"  MODEL_ID={args.model_id}")
    print("  LOCAL_FILES_ONLY=1   # after first successful download")
    print("  # worker passes local_files_only=True into from_pretrained (env alone is not enough)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
