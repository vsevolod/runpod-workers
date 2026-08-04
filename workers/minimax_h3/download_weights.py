#!/usr/bin/env python3
"""Bootstrap MiniMax-H3 t2va weights onto a Network Volume (or local dir).

Run on a Pod with the volume mounted (not inside a serverless job):

    python download_weights.py --output /runpod-volume/minimax_h3

Does NOT bake weights into the Docker image (~144 GB t2va half).

Packs:
  * ``official`` (default) — full MiniMaxAI t2va half (~144 GB)
  * ``hybrid_spike`` — Comfy non-pruned int8 DiT + official TE/VAE +
    ``transformer/config.json`` only (Level-1 R&D; not production default)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "MiniMaxAI/MiniMax-H3"
COMFY_REPO = "Comfy-Org/MiniMax-H3"
COMFY_DIT_NON_PRUNED = "diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors"
COMFY_DIT_PRUNED = "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"

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

# MiniMaxAI snapshot for hybrid: TE/VAE/tokenizers + DiT *config only* (offline G4).
ALLOW_PATTERNS_HYBRID_TE_VAE = [
    "text_encoder/*",
    "text_encoder/**",
    "tokenizer/*",
    "tokenizer/**",
    "processor/*",
    "processor/**",
    "vae/*",
    "vae/**",
    "audio_vae/*",
    "audio_vae/**",
    "scheduler/*",
    "scheduler/**",
    "audio_scheduler/*",
    "audio_scheduler/**",
    "modular_model_index.json",
    "model_index.json",
    # Required for meta from_config / G4 without Hub network:
    "transformer/config.json",
    # Do NOT include transformer/*.safetensors or transformer/**
]


def _snapshot_download(*args, **kwargs):
    """Indirection so unit tests can mock without importing huggingface_hub."""
    from huggingface_hub import snapshot_download

    return snapshot_download(*args, **kwargs)


def _hf_hub_download(*args, **kwargs):
    """Indirection so unit tests can mock without importing huggingface_hub."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(*args, **kwargs)


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
        "--pack",
        choices=("official", "hybrid_spike"),
        default="official",
        help="Weight pack: official t2va half, or Level-1 hybrid_spike",
    )
    parser.add_argument(
        "--also-fetch-pruned-for-g0",
        action="store_true",
        help="hybrid_spike only: also download pruned DiT for G0 inspect (not used for load)",
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

    if args.pack == "hybrid_spike":
        return _main_hybrid(args, out)
    return _main_official(args, out)


def _main_official(args: argparse.Namespace, out: Path) -> int:
    print(f"Pack: official")
    print(f"Repo: {args.repo}")
    print(f"Output: {out}")
    print(f"Allow patterns ({len(ALLOW_PATTERNS_T2VA)}):")
    for p in ALLOW_PATTERNS_T2VA:
        print(f"  - {p}")

    if args.dry_run:
        print("Dry-run: not calling snapshot_download.")
        return 0

    out.mkdir(parents=True, exist_ok=True)
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


def _main_hybrid(args: argparse.Namespace, out: Path) -> int:
    print("Pack: hybrid_spike (Level-1 R&D — not production default)")
    print(f"Comfy DiT repo: {COMFY_REPO}")
    print(f"  non-pruned: {COMFY_DIT_NON_PRUNED}")
    if args.also_fetch_pruned_for_g0:
        print(f"  pruned (G0 only): {COMFY_DIT_PRUNED}")
    print(f"Official TE/VAE repo: {args.repo}")
    print(f"Output: {out}")
    print(f"Allow patterns hybrid TE/VAE ({len(ALLOW_PATTERNS_HYBRID_TE_VAE)}):")
    for p in ALLOW_PATTERNS_HYBRID_TE_VAE:
        print(f"  - {p}")
    print("Approx size: DiT ~34 GB + TE/VAE ~78 GB + config ≪1 MB ≈ 112 GB")

    if args.dry_run:
        print("Dry-run: not calling hf_hub_download / snapshot_download.")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    comfy_dir = out / "comfy"
    comfy_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    print(f"Download Comfy non-pruned DiT -> {comfy_dir} …")
    dit_path = _hf_hub_download(
        repo_id=COMFY_REPO,
        filename=COMFY_DIT_NON_PRUNED,
        local_dir=str(comfy_dir),
        token=args.token,
    )
    print(f"  -> {dit_path}")

    if args.also_fetch_pruned_for_g0:
        print(f"Download Comfy pruned DiT (G0 inspect) -> {comfy_dir} …")
        pruned_path = _hf_hub_download(
            repo_id=COMFY_REPO,
            filename=COMFY_DIT_PRUNED,
            local_dir=str(comfy_dir),
            token=args.token,
        )
        print(f"  -> {pruned_path}")

    print(f"Snapshot official TE/VAE + transformer/config.json -> {out} …")
    path = _snapshot_download(
        args.repo,
        local_dir=str(out),
        token=args.token,
        allow_patterns=list(ALLOW_PATTERNS_HYBRID_TE_VAE),
    )
    print(f"  -> {path}")

    print("Done.")
    print()
    print("Hybrid spike layout:")
    print(f"  {out}/comfy/diffusion_models/…_int8_convrot.safetensors")
    print(f"  {out}/transformer/config.json  # offline G4 meta-init")
    print(f"  {out}/text_encoder/ …")
    print(f"  {out}/vae/ …")
    print("G0 inspect:")
    print(
        "  PYTHONPATH=. python tools/spike_inspect_comfy_dit.py "
        f"{out}/comfy/{COMFY_DIT_NON_PRUNED}"
    )
    print("G4 forward (after wiring):")
    print(
        "  PYTHONPATH=. python tools/spike_dit_forward.py "
        f"--dit {out}/comfy/{COMFY_DIT_NON_PRUNED} --model-dir {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
