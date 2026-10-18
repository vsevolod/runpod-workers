#!/usr/bin/env python3
"""Download Krea 2 DiT (FP8 or INT8 ConvRot) + TE/VAE before inference starts.

The container entrypoint runs this on every fresh worker. For a manual download:

    python download_weights.py --output /models/krea2 --hf-home /models/hf
    python download_weights.py --output /models/krea2 --hf-home /models/hf --quant int8_convrot

Does NOT bake weights into the Docker image. Both the DiT files and HF caches
can live on container disk; an attached volume is only needed for extra LoRAs.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

workers_dir = Path(__file__).resolve().parents[1]
if (workers_dir / "shared").is_dir():
    sys.path.insert(0, str(workers_dir))
from shared.model_download import ModelDownloader, add_download_arguments, link_or_copy
from shared.prefetch import ComponentPublisher

# FP8 pack (default / legacy)
DIT_REPO_FP8 = "AlperKTS/Krea2_FP8"
DIT_FILE_FP8 = "krea2_turbo_fp8.safetensors"

# INT8 ConvRot pack (Comfy-Org native int8_tensorwise)
DIT_REPO_INT8 = "Comfy-Org/Krea-2"
DIT_FILE_INT8 = "diffusion_models/krea2_turbo_int8_convrot.safetensors"
DIT_FILE_INT8_FLAT = "krea2_turbo_int8_convrot.safetensors"

VAE_FILE = "qwen_image_vae.safetensors"
TEXT_ENCODER_ID = "Qwen/Qwen3-VL-4B-Instruct"
VAE_REPO = "Qwen/Qwen-Image"
EDIT_LORA_REPO = "conradlocke/krea2-identity-edit"
EDIT_LORA_REVISION = "f33b26291c640e8ec50b0da420bfee3c9a660daa"
EDIT_LORA_FILES = ("krea2_identity_edit_v2.safetensors", "krea2_general_edit_v2.safetensors")


def _load_dit_quant_module():
    """Load dit_quant.py without importing krea2_infer package (heavy TE deps)."""
    path = Path(__file__).resolve().parent / "krea2_infer" / "dit_quant.py"
    spec = importlib.util.spec_from_file_location("_krea2_dit_quant_for_download", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load dit_quant from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_download_quant(value: str | None) -> tuple[bool, bool]:
    """Map --quant / DIT_QUANT to (want_fp8, want_int8).

    Shares aliases with ``resolve_dit_quant``; extra value ``both`` downloads both.
    """
    raw = (value if value is not None else "fp8").strip().lower()
    if raw == "both":
        return True, True
    dq = _load_dit_quant_module()
    mode = dq.resolve_dit_quant(raw)
    if mode == dq.MODE_INT8_CONVROT:
        return False, True
    if mode == dq.MODE_FP8:
        return True, False
    raise ValueError(f"Unsupported download quant mode: {value!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_download_arguments(parser, default_output="./models/krea2")
    parser.add_argument(
        "--quant",
        default=os.environ.get("DIT_QUANT", "fp8"),
        # No strict choices: aliases match runtime resolve_dit_quant + "both".
        help="Which DiT quant pack to download (default: DIT_QUANT or fp8). "
        "Same aliases as DIT_QUANT (fp8, int8_convrot, int8, int8_tensorwise, …); "
        "'both' fetches FP8 and INT8 ConvRot.",
    )
    parser.add_argument(
        "--skip-dit",
        action="store_true",
        help="Skip DiT download",
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
    args = parser.parse_args(argv)

    out = Path(args.output).expanduser().resolve()
    print(f"MODEL_DIR={out}")
    try:
        want_fp8, want_int8 = resolve_download_quant(args.quant)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(f"download quant={args.quant!r} -> fp8={want_fp8} int8_convrot={want_int8}")
    publisher = ComponentPublisher(args.prefetch_state_dir)
    with ModelDownloader(out, hf_home=args.hf_home, token=args.token,
                         pin_revisions=bool(args.prefetch_state_dir)) as downloads:
        if not args.skip_dit:
            if want_fp8:
                dit = downloads.file(DIT_REPO_FP8, DIT_FILE_FP8)
            if want_int8:
                dit = downloads.file(DIT_REPO_INT8, DIT_FILE_INT8)
                link_or_copy(dit, out / DIT_FILE_INT8_FLAT)
            publisher.ready("dit", dit)

        if not args.skip_vae_file:
            # Community single-file is optional; the clean HF VAE is downloaded below.
            try:
                downloads.file(DIT_REPO_FP8, VAE_FILE)
            except Exception as err:
                print(f"  AlperKTS VAE not available ({err}); will use {VAE_REPO} cache only")

        if not args.skip_text_encoder:
            vae = downloads.snapshot(VAE_REPO, allow_patterns=["vae/*", "VAE/*", "*.json"])
            publisher.ready("vae", vae)
            text_encoder_id = os.environ.get("TEXT_ENCODER_ID", TEXT_ENCODER_ID)
            if Path(text_encoder_id).is_dir():
                print(f"Using existing local text encoder: {Path(text_encoder_id).resolve()}")
                encoder = Path(text_encoder_id).resolve()
            else:
                encoder = downloads.snapshot(text_encoder_id)
            publisher.ready("text_encoder", encoder)
        edit_loras = out / "edit_loras"
        edit_loras.mkdir(exist_ok=True)
        for filename in EDIT_LORA_FILES:
            adapter = downloads.file(EDIT_LORA_REPO, filename, revision=EDIT_LORA_REVISION)
            link_or_copy(adapter, edit_loras / filename)
        publisher.ready("edit_loras", edit_loras)
    publisher.ready("all", out)
    print()
    print("Model directory layout:")
    if want_fp8:
        print(f"  {out}/{DIT_FILE_FP8}")
    if want_int8:
        print(f"  {out}/{DIT_FILE_INT8_FLAT}")
    print(f"  {out}/{VAE_FILE}   # optional if HF VAE cache is warm")
    print("Env on endpoint:")
    print(f"  MODEL_DIR={out}")
    if want_int8 and not want_fp8:
        print("  DIT_QUANT=int8_convrot")
    elif want_int8 and want_fp8:
        print("  # FP8 default: unset DIT_QUANT")
        print("  # INT8 path:   DIT_QUANT=int8_convrot")
    if args.hf_home:
        print(f"  HF_HOME={args.hf_home}")
        print("  LOCAL_FILES_ONLY=1   # after first successful download")
    return 0


if __name__ == "__main__":
    sys.exit(main())
