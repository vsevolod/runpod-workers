"""Where the worker loads Qwen-Image-2.1 from, and how it places weights on GPU."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MODEL_ID = "Qwen/Qwen-Image-2.1"
DEFAULT_MODEL_DIR = "/runpod-volume/qwen_image_21"
LORA_REPO_ID = "Viggle/Qwen-Image-2.1-viggle-turbo"
LORA_WEIGHT_NAME = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
LORA_DIR_NAME = "loras"

_OFFLOAD_ALIASES = {
    "cpu": "cpu",
    "model": "cpu",
    "sequential": "sequential",
    "none": "none",
    "cuda": "none",
    "gpu": "none",
}


def resolve_model_source(model_dir: str | None = None, model_id: str | None = None) -> str:
    """Prefer a local Diffusers snapshot (model_index.json) over the Hub id."""
    directory = Path(
        model_dir if model_dir is not None else os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)
    )
    if (directory / "model_index.json").is_file():
        return str(directory)
    if model_id is not None:
        return model_id
    return os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)


def resolve_offload(value: str | None = None) -> str:
    """Return ``cpu``, ``sequential``, or ``none``.

    ``cpu`` is model-level offload (text encoder, then transformer, then VAE).
    That is the 24 GB path. ``none`` keeps the pipeline resident on CUDA.
    """
    raw = value if value is not None else os.environ.get("OFFLOAD", "cpu")
    key = str(raw).strip().lower()
    try:
        return _OFFLOAD_ALIASES[key]
    except KeyError as err:
        known = ", ".join(sorted(_OFFLOAD_ALIASES))
        raise ValueError(f"OFFLOAD must be one of {known} (got {value!r})") from err


def resolve_lora_source(
    model_dir: str | None = None,
    *,
    local_files_only: bool | None = None,
) -> tuple[str, str, bool]:
    """Return ``(directory or repo id, weight filename, local_files_only)``.

    A file already on the volume wins. ``LOCAL_FILES_ONLY`` with no file is an
    error. Otherwise the Hub repo is returned and Diffusers downloads the adapter.
    """
    files_only = parse_local_files_only() if local_files_only is None else local_files_only
    directory = Path(
        model_dir if model_dir is not None else os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)
    )
    for folder in (directory / LORA_DIR_NAME, directory):
        if (folder / LORA_WEIGHT_NAME).is_file():
            return str(folder), LORA_WEIGHT_NAME, True
    if files_only:
        raise FileNotFoundError(
            f"Viggle turbo LoRA {LORA_WEIGHT_NAME} is not under {directory}. "
            "Re-run download_weights.py so it fetches the adapter into loras/."
        )
    return LORA_REPO_ID, LORA_WEIGHT_NAME, False


def parse_local_files_only(value: str | None = None) -> bool:
    raw = value if value is not None else os.environ.get("LOCAL_FILES_ONLY", "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}
