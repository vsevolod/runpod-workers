"""Normalize T2V job input beyond schema defaults."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import Any, Mapping

from h3_infer.canvas import validate_canvas
from h3_infer.duration import (
    output_duration_sec,
    snap_num_frames,
    validate_requested_duration,
)

DEFAULT_WIDTH = 864
DEFAULT_HEIGHT = 480
MAX_PROMPT_CHARS = 8000


class RequestError(ValueError):
    """Safe client-facing validation error."""


@dataclass(frozen=True)
class T2VRequest:
    prompt: str
    width: int
    height: int
    requested_duration: float
    length: int
    seed: int
    output_duration: float


def normalize_t2v_input(raw: Mapping[str, Any]) -> T2VRequest:
    """Validate and normalize a T2V input dict into ``T2VRequest``."""
    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise RequestError("prompt must be a non-empty string")
    prompt = prompt.strip()
    if len(prompt) > MAX_PROMPT_CHARS:
        raise RequestError(
            f"prompt exceeds {MAX_PROMPT_CHARS} characters (got {len(prompt)})"
        )

    width = raw.get("width", DEFAULT_WIDTH)
    height = raw.get("height", DEFAULT_HEIGHT)
    if not isinstance(width, int) or isinstance(width, bool):
        raise RequestError(f"width must be int, got {type(width).__name__}")
    if not isinstance(height, int) or isinstance(height, bool):
        raise RequestError(f"height must be int, got {type(height).__name__}")

    try:
        validate_canvas(width, height)
    except ValueError as err:
        raise RequestError(str(err)) from err

    duration_raw = raw.get("duration", 5.0)
    try:
        requested = validate_requested_duration(duration_raw)
    except ValueError as err:
        raise RequestError(str(err)) from err

    length = snap_num_frames(requested)
    out_dur = output_duration_sec(length)

    seed_raw = raw.get("seed", None)
    if seed_raw is None or seed_raw == -1:
        seed = int.from_bytes(secrets.token_bytes(4), "big")
    else:
        if not isinstance(seed_raw, int) or isinstance(seed_raw, bool) or seed_raw < 0:
            raise RequestError("seed must be an int >= -1")
        seed = int(seed_raw)

    return T2VRequest(
        prompt=prompt,
        width=width,
        height=height,
        requested_duration=requested,
        length=length,
        seed=seed,
        output_duration=out_dur,
    )


def model_dir_from_env() -> str:
    return os.environ.get("MODEL_DIR", "/runpod-volume/minimax_h3")
