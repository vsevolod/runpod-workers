#!/usr/bin/env python3
"""Materialize Comfy models tree from RunPod HF cache or legacy Network Volume.

Always writes symlinks under /models (hard-coded). Never copies weight bytes.
Never downloads from Hugging Face at boot.

CLI::

    python -u /app/model_store.py   # exit 0 on success, non-zero on failure
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Shared with download_weights (operator CLI for offline volume fill).
try:
    from download_weights import WEIGHTS
except ImportError:  # pragma: no cover — same directory on PYTHONPATH
    WEIGHTS = [
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

WEIGHT_RELS: list[str] = [rel for rel, _ in WEIGHTS]

# Hard-coded materialize root (v1 — not configurable via env).
COMFY_MODELS = Path("/models")

DEFAULT_MODEL_NAME = "Comfy-Org/MiniMax-H3"
DEFAULT_HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"

# Ops candidate roots when MODEL_DIR is unset or incomplete (legacy volume).
_VOLUME_CANDIDATES = (
    "/runpod-volume/minimax_h3_comfy",
    "/workspace/minimax_h3_comfy",
    "/runpod-volume",
    "/workspace",
)


class ModelStoreError(Exception):
    """Resolvable bootstrap failure (cache/volume missing or ambiguous)."""


def log(msg: str) -> None:
    print(f"[ModelStore] {msg}", flush=True)


def _parse_model_id(model_id: str) -> tuple[str, str]:
    if not model_id or not model_id.strip():
        raise ModelStoreError("MODEL_NAME is empty; set org/name matching endpoint Model field")
    model_id = model_id.strip()
    if ":" in model_id:
        raise ModelStoreError(
            f"MODEL_NAME pin syntax not supported in v1 (got {model_id!r}); "
            "use plain org/name (main branch only)"
        )
    if "/" not in model_id or model_id.count("/") != 1:
        raise ModelStoreError(
            f"MODEL_NAME must be org/name (got {model_id!r})"
        )
    org, name = model_id.split("/", 1)
    if not org or not name:
        raise ModelStoreError(f"MODEL_NAME must be org/name (got {model_id!r})")
    return org, name


def resolve_snapshot_path(model_id: str, cache_root: Path | str) -> Path:
    """Resolve HF hub snapshot dir for model_id under cache_root.

    1. refs/main → snapshots/<hash> if directory
    2. else exactly one snapshots/* subdir
    3. else fail with candidate list
    """
    org, name = _parse_model_id(model_id)
    cache_root = Path(cache_root)
    model_root = cache_root / f"models--{org}--{name}"

    if not model_root.is_dir():
        raise ModelStoreError(
            f"HF model cache not found: {model_root} "
            f"(HF_CACHE_ROOT={cache_root}, MODEL_NAME={org}/{name}). "
            "Set endpoint Model field / MODEL_NAME, wait for store fill, "
            "or attach legacy volume with MODEL_DIR."
        )

    snapshots_dir = model_root / "snapshots"
    refs_main = model_root / "refs" / "main"

    if refs_main.is_file():
        snap_hash = refs_main.read_text(encoding="utf-8").strip()
        if not snap_hash:
            raise ModelStoreError(f"refs/main is empty under {model_root}")
        snap = snapshots_dir / snap_hash
        if snap.is_dir():
            return snap
        raise ModelStoreError(
            f"refs/main points to {snap_hash!r} but snapshot dir missing: {snap} "
            f"(cache_root={cache_root}, model_root={model_root})"
        )

    if not snapshots_dir.is_dir():
        raise ModelStoreError(
            f"No refs/main and no snapshots/ under {model_root} "
            f"(HF_CACHE_ROOT={cache_root})"
        )

    candidates = sorted(p.name for p in snapshots_dir.iterdir() if p.is_dir())
    if len(candidates) == 1:
        return snapshots_dir / candidates[0]
    if not candidates:
        raise ModelStoreError(
            f"Empty snapshots/ under {model_root} (HF_CACHE_ROOT={cache_root})"
        )
    raise ModelStoreError(
        f"Ambiguous snapshots under {model_root}: {candidates} "
        f"(no refs/main; HF_CACHE_ROOT={cache_root}). "
        "Refuse to pick lexicographically."
    )


def find_weight_sources(root: Path) -> dict[str, Path]:
    """Locate four weights under root (top-level or models/ layout).

    Accepts:
      root/diffusion_models/...
      root/models/diffusion_models/...
    """
    root = Path(root)
    found: dict[str, Path] = {}
    missing: list[str] = []

    for rel in WEIGHT_RELS:
        candidates = [
            root / rel,
            root / "models" / rel,
        ]
        hit = next((c for c in candidates if c.is_file()), None)
        if hit is None:
            missing.append(rel)
        else:
            found[rel] = hit.resolve()

    if missing:
        raise ModelStoreError(
            f"MISSING weight(s) under {root}: {missing}. "
            "Partial/incomplete store, wrong MODEL_NAME/snapshot, or dual-layout miss."
        )
    return found


def materialize_comfy_models(
    sources: dict[str, Path],
    dest: Path = COMFY_MODELS,
) -> None:
    """Symlink each weight into dest. Never copy bytes. Idempotent."""
    dest = Path(dest)
    for rel, src in sources.items():
        src = Path(src).resolve()
        if not src.is_file():
            raise ModelStoreError(f"source missing for {rel}: {src}")
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(src)
        if not target.is_symlink():
            raise ModelStoreError(f"failed to create symlink at {target}")
        log(f"linked {rel} -> {src}")


def _try_cache_sources(model_name: str, cache_root: Path) -> dict[str, Path] | None:
    """Return weight sources from HF cache, or None if cache not usable."""
    try:
        snap = resolve_snapshot_path(model_name, cache_root)
    except ModelStoreError as e:
        log(f"cache not usable: {e}")
        return None
    log(f"Using snapshot: {snap}")
    try:
        return find_weight_sources(snap)
    except ModelStoreError as e:
        log(f"snapshot incomplete: {e}")
        return None


def _volume_roots(model_dir: str | None) -> list[Path]:
    roots: list[Path] = []
    if model_dir:
        roots.append(Path(model_dir))
    for c in _VOLUME_CANDIDATES:
        p = Path(c)
        if p not in roots:
            roots.append(p)
    return roots


def _try_volume_sources(model_dir: str | None) -> dict[str, Path] | None:
    """Return weight sources from legacy volume models/ tree."""
    for root in _volume_roots(model_dir):
        models = root / "models"
        if not models.is_dir():
            continue
        try:
            sources = find_weight_sources(models)
        except ModelStoreError:
            # Also accept if find looks under models/models — try root as models parent
            try:
                sources = find_weight_sources(root)
            except ModelStoreError as e:
                log(f"volume candidate incomplete {root}: {e}")
                continue
        log(f"legacy volume root: {root}")
        return sources
    return None


def prepare_models(dest: Path = COMFY_MODELS) -> str:
    """Resolve cache or volume and materialize into dest. Returns source tag."""
    model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
    cache_root = Path(os.environ.get("HF_CACHE_ROOT", DEFAULT_HF_CACHE_ROOT))
    model_dir = os.environ.get("MODEL_DIR")  # may be None

    log(f"MODEL_NAME={model_name}")
    log(f"HF_CACHE_ROOT={cache_root}")
    if model_dir:
        log(f"MODEL_DIR={model_dir}")

    sources = _try_cache_sources(model_name, cache_root)
    source_tag = "cache"
    if sources is None:
        sources = _try_volume_sources(model_dir)
        source_tag = "volume"

    if sources is None:
        raise ModelStoreError(
            "No usable model source. "
            f"Configure endpoint Model / MODEL_NAME={model_name!r} with HF cache at "
            f"{cache_root}, or attach Network Volume and set MODEL_DIR to a root "
            "containing models/ with the four T2V weights "
            f"({', '.join(WEIGHT_RELS)}). "
            "Runtime HF download is not supported in v1."
        )

    materialize_comfy_models(sources, dest=dest)
    log(f"source={source_tag}")
    return source_tag


def main(argv: list[str] | None = None) -> int:
    del argv  # CLI takes no args; env-driven
    try:
        prepare_models(dest=COMFY_MODELS)
    except ModelStoreError as e:
        log(f"ERROR: {e}")
        return 1
    except Exception as e:  # pragma: no cover — unexpected
        log(f"ERROR: unexpected {type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
