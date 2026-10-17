"""Where the worker loads Qwen-Image-2.1 from, and how it places weights on GPU."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

DEFAULT_MODEL_ID = "Qwen/Qwen-Image-2.1-Turbo"
DEFAULT_MODEL_DIR = "/models/qwen_image_21_turbo"
LORA_DIR_NAME = "loras"
TURBO_STEPS = 8
BASE_STEPS = 40
MODEL_IDS = {"turbo": DEFAULT_MODEL_ID, "base": "Qwen/Qwen-Image-2.1"}
TEXTURE_FIX_VAE_ID = "madebyollin/texture-fix-vae-for-qwen-image-2.1"
TEXTURE_FIX_VAE_DIR = "texture_fix_vae"
_OFFLOAD_ALIASES = {
    "cpu": "cpu",
    "model": "cpu",
    "sequential": "sequential",
    "none": "none",
    "cuda": "none",
    "gpu": "none",
}


def resolve_model_variant(value: str = "turbo") -> str:
    if value not in MODEL_IDS:
        raise ValueError("model_variant must be 'base' or 'turbo'")
    return value


def resolve_model_directory(model_variant: str = "turbo") -> Path:
    variant = resolve_model_variant(model_variant)
    turbo = Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))
    if variant == "turbo":
        return turbo
    base = Path(os.environ.get("BASE_MODEL_DIR", str(turbo.parent / "qwen_image_21_base")))
    if base.resolve() == turbo.resolve():
        raise ValueError("BASE_MODEL_DIR and MODEL_DIR must be different directories")
    return base


def resolve_model_id(model_variant: str = "turbo") -> str:
    variant = resolve_model_variant(model_variant)
    key = "MODEL_ID" if variant == "turbo" else "BASE_MODEL_ID"
    return os.environ.get(key) or MODEL_IDS[variant]


def resolve_vae_directory() -> Path:
    return Path(os.environ.get("QWEN_VAE_DIR") or
                resolve_model_directory("turbo").parent / "qwen_image_21_shared" / TEXTURE_FIX_VAE_DIR)


def resolve_vae_variant(value: str | None = None) -> str:
    raw = value if value is not None else os.environ.get("QWEN_VAE", "texture-fix")
    variant = raw.strip().lower()
    if variant not in {"texture-fix", "original"}:
        raise ValueError(f"QWEN_VAE must be texture-fix or original (got {variant!r})")
    return variant


def is_obsolete_accelerator_lora(filename: str) -> bool:
    """Exclude obsolete Viggle accelerators left in a shared LoRA directory."""
    return Path(filename).name.startswith("Qwen-Image-2.1-viggle-turbo")


def resolve_model_source(model_dir: str | None = None, model_id: str | None = None,
                         model_variant: str = "turbo") -> str:
    """Prefer a local Diffusers snapshot (model_index.json) over the Hub id."""
    directory = Path(
        model_dir if model_dir is not None else resolve_model_directory(model_variant)
    )
    if (directory / "model_index.json").is_file():
        validate_model_snapshot(directory, model_variant)
        return str(directory)
    if model_id is not None:
        return model_id
    return resolve_model_id(model_variant)


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


def validate_turbo_config(config, source: str) -> None:
    """Require a saved eight-step grid; never silently use the base model schedule."""
    sigmas = config.get("sample_sigmas")
    valid = (
        isinstance(sigmas, (list, tuple))
        and len(sigmas) == TURBO_STEPS
        and all(
            isinstance(sigma, (int, float)) and not isinstance(sigma, bool)
            and math.isfinite(sigma) and 0 < sigma <= 1
            for sigma in sigmas
        )
        and all(left > right for left, right in zip(sigmas, sigmas[1:]))
    )
    if not valid:
        raise ValueError(
            f"Qwen-Image-2.1-Turbo requires a saved {TURBO_STEPS}-step sample_sigmas "
            f"schedule (source: {source}). Download the official Turbo checkpoint "
            "into a separate directory and update MODEL_DIR/MODEL_ID; "
            "the old base-model/Viggle snapshot cannot be reused."
        )


def validate_turbo_snapshot(directory: Path | str) -> None:
    validate_model_snapshot(directory, "turbo")


def validate_model_config(config, source: str, model_variant: str = "turbo") -> None:
    """Validate the serialized model_index.json, including its pipeline class."""
    variant = resolve_model_variant(model_variant)
    if not isinstance(config, dict) or config.get("_class_name") != "QwenImage21Pipeline":
        raise ValueError(f"Expected Qwen-Image-2.1-{variant.title()} QwenImage21Pipeline in {source}")
    validate_sampling_config(config, source, variant)


def validate_sampling_config(config, source: str, model_variant: str = "turbo") -> None:
    """Check runtime sampling settings; Diffusers omits serialized class metadata."""
    variant = resolve_model_variant(model_variant)
    if variant == "turbo":
        validate_turbo_config(config, source)
    elif config.get("sample_sigmas") is not None:
        raise ValueError(f"Base model must not contain a Turbo sample_sigmas schedule (source: {source})")


def validate_model_snapshot(directory: Path | str, model_variant: str = "turbo") -> None:
    """Validate local metadata before loading weights or overwriting a snapshot."""
    index = Path(directory) / "model_index.json"
    try:
        config = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise ValueError(f"Invalid Qwen-Image-2.1 {model_variant.title()} model_index.json: {index}") from err
    validate_model_config(config, str(directory), model_variant)


def parse_local_files_only(value: str | None = None) -> bool:
    raw = value if value is not None else os.environ.get("LOCAL_FILES_ONLY", "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def parse_disable_mmap(value: str | None = None) -> bool:
    """Keep mmap's lower RAM usage by default; eager loading is an opt-in experiment."""
    raw = value if value is not None else os.environ.get("QWEN_DISABLE_MMAP", "0")
    key = str(raw).strip().lower()
    if key in {"1", "true", "yes", "on"}:
        return True
    if key in {"0", "false", "no", "off"}:
        return False
    raise ValueError("QWEN_DISABLE_MMAP must be 1/true/yes/on or 0/false/no/off")
