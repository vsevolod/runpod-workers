"""DiT quantization mode selection (FP8 storage vs INT8 ConvRot compute)."""

from __future__ import annotations

import os
from typing import Sequence

# Canonical modes used by load_pipeline / load_dit.
MODE_FP8 = "fp8"
MODE_INT8_CONVROT = "int8_convrot"

_FP8_ALIASES = frozenset({"fp8", "default", "bf16", "float8"})
_INT8_ALIASES = frozenset(
    {
        "int8_convrot",
        "int8",
        "int8_tensorwise",
        "int8-convrot",
        "convrot",
    }
)

DIT_CANDIDATES_FP8: tuple[str, ...] = (
    "krea2_turbo_fp8.safetensors",
    "krea2_turbo.safetensors",
    "oss_turbo.safetensors",
)

DIT_CANDIDATES_INT8_CONVROT: tuple[str, ...] = (
    "krea2_turbo_int8_convrot.safetensors",
    "Krea2-Turbo-int8-ConvRot.safetensors",
    "krea2_turbo_convrot_int8.safetensors",
)


def resolve_dit_quant(value: str | None = None) -> str:
    """Return canonical DIT quant mode from explicit value or DIT_QUANT env.

    Defaults to ``fp8`` (current production path).
    """
    if value is None:
        raw = os.environ.get("DIT_QUANT", MODE_FP8)
    else:
        raw = value
    key = (raw or MODE_FP8).strip().lower()
    if key in _FP8_ALIASES or key == "":
        return MODE_FP8
    if key in _INT8_ALIASES:
        return MODE_INT8_CONVROT
    raise ValueError(
        f"Unsupported DIT_QUANT={raw!r}. Expected one of: "
        f"{MODE_FP8}, {MODE_INT8_CONVROT} (aliases: int8, int8_tensorwise)."
    )


def dit_candidates_for(mode: str) -> Sequence[str]:
    mode = resolve_dit_quant(mode)
    if mode == MODE_INT8_CONVROT:
        return DIT_CANDIDATES_INT8_CONVROT
    return DIT_CANDIDATES_FP8
