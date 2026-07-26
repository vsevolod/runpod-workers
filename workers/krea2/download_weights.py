#!/usr/bin/env python3
"""Bootstrap Krea 2 DiT (FP8 or INT8 ConvRot) + TE/VAE caches onto a Network Volume.

Run on a Pod with the volume mounted, or any machine with disk + HF access:

    python download_weights.py --output /runpod-volume/krea2
    python download_weights.py --output /runpod-volume/krea2 --quant int8_convrot

Does NOT bake weights into the Docker image (prefer volume for ~13–18 GB DiT/VAE
plus HF text-encoder cache).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

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
        "--quant",
        default=os.environ.get("DIT_QUANT", "fp8"),
        # No strict choices: aliases match runtime resolve_dit_quant + "both".
        help="Which DiT quant pack to download (default: DIT_QUANT or fp8). "
        "Same aliases as DIT_QUANT (fp8, int8_convrot, int8, int8_tensorwise, …); "
        "'both' fetches FP8 and INT8 ConvRot.",
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
    try:
        want_fp8, want_int8 = resolve_download_quant(args.quant)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(f"download quant={args.quant!r} -> fp8={want_fp8} int8_convrot={want_int8}")

    if not args.skip_dit:
        if want_fp8:
            print(f"Downloading DiT {DIT_REPO_FP8}/{DIT_FILE_FP8} …")
            dit = _hf_download(DIT_REPO_FP8, DIT_FILE_FP8, out, args.token)
            print(f"  -> {dit}")
        if want_int8:
            print(f"Downloading DiT {DIT_REPO_INT8}/{DIT_FILE_INT8} …")
            dit = _hf_download(DIT_REPO_INT8, DIT_FILE_INT8, out, args.token)
            print(f"  -> {dit}")
            # Prefer a flat filename next to FP8 for simple DIT_PATH / candidates.
            flat = out / DIT_FILE_INT8_FLAT
            nested = out / DIT_FILE_INT8
            if nested.is_file() and not flat.is_file():
                try:
                    flat.hardlink_to(nested)
                    print(f"  hardlink {flat} -> {nested}")
                except OSError:
                    import shutil

                    shutil.copy2(nested, flat)
                    print(f"  copied {nested} -> {flat}")

    if not args.skip_vae_file:
        # Prefer community single-file if present on the FP8 repo; otherwise HF repo.
        try:
            print(f"Trying VAE {DIT_REPO_FP8}/{VAE_FILE} …")
            vae = _hf_download(DIT_REPO_FP8, VAE_FILE, out, args.token)
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
