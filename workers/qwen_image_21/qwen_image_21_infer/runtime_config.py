"""Where the worker loads Qwen-Image-2.1 from, and how it places weights on GPU."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MODEL_ID = "Qwen/Qwen-Image-2.1"
DEFAULT_MODEL_DIR = "/runpod-volume/qwen_image_21"

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


def parse_local_files_only(value: str | None = None) -> bool:
    raw = value if value is not None else os.environ.get("LOCAL_FILES_ONLY", "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}
