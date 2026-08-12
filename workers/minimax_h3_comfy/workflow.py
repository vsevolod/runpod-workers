"""Load frozen API workflow and inject product fields (exact node ids from PINS.md)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

# Inject map — literals must match PINS.md / workflows/t2va_api.json
PROMPT_NODE = "104"
PROMPT_KEY = "prompt"
WIDTH_KEY = "width"
HEIGHT_KEY = "height"

DURATION_NODE = "111"
DURATION_KEY = "value"

SEED_NODE = "15"
SEED_KEY = "noise_seed"

UNET_NODE = "6"
UNET_NAME_KEY = "unet_name"
SAVE_VIDEO_NODE = "92"

# Runtime LoadImage nodes for I2V / FL2V (not present in frozen t2va_api.json)
FIRST_LOAD_NODE = "200"
LAST_LOAD_NODE = "201"
FIRST_FRAME_KEY = "first_frame"
LAST_FRAME_KEY = "last_frame"

DEFAULT_WIDTH = 864
DEFAULT_HEIGHT = 480
DEFAULT_DURATION = 5.0

# H3 native canvas (official docs): short edge 768, max 768×1344, multiple of 32
MAX_SHORT_EDGE = 768
MAX_LONG_EDGE = 1344
CANVAS_MULTIPLE = 32


def template_path() -> Path:
    return Path(__file__).resolve().parent / "workflows" / "t2va_api.json"


def load_workflow(path: Path | None = None) -> dict[str, Any]:
    p = path or template_path()
    if not p.is_file():
        raise FileNotFoundError(f"API workflow missing: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Workflow must be a non-empty API dict: {p}")
    return data


def validate_canvas(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive, got {width}x{height}")
    if width % CANVAS_MULTIPLE != 0 or height % CANVAS_MULTIPLE != 0:
        raise ValueError(
            f"width/height must be multiples of {CANVAS_MULTIPLE}, got {width}x{height}"
        )
    short_e, long_e = sorted((width, height))
    if short_e > MAX_SHORT_EDGE:
        raise ValueError(
            f"short edge {short_e} exceeds H3 max short edge {MAX_SHORT_EDGE}"
        )
    if long_e > MAX_LONG_EDGE:
        raise ValueError(
            f"long edge {long_e} exceeds H3 max long edge {MAX_LONG_EDGE}"
        )


def inject_product(
    workflow: dict[str, Any],
    *,
    prompt: str,
    width: int,
    height: int,
    duration: float,
    seed: int,
    first_image_name: str | None = None,
    last_image_name: str | None = None,
) -> dict[str, Any]:
    """Deep-copy workflow; set inject-map fields; optional I2V LoadImage wiring.

    first_image_name / last_image_name are basenames under Comfy input/ for LoadImage.
    last_image_name requires first_image_name.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    validate_canvas(int(width), int(height))

    if last_image_name and not first_image_name:
        raise ValueError("last_image_name requires first_image_name")
    if first_image_name is not None and (
        not isinstance(first_image_name, str) or not first_image_name.strip()
    ):
        raise ValueError("first_image_name must be a non-empty basename")
    if last_image_name is not None and (
        not isinstance(last_image_name, str) or not last_image_name.strip()
    ):
        raise ValueError("last_image_name must be a non-empty basename")

    out = copy.deepcopy(workflow)

    def _set(node_id: str, key: str, value: Any) -> None:
        if node_id not in out:
            raise KeyError(f"workflow missing node {node_id!r}")
        node = out[node_id]
        if not isinstance(node, dict) or "inputs" not in node:
            raise KeyError(f"node {node_id!r} has no inputs")
        if key not in node["inputs"]:
            raise KeyError(f"node {node_id!r} missing input {key!r}")
        node["inputs"][key] = value

    _set(PROMPT_NODE, PROMPT_KEY, prompt)
    _set(PROMPT_NODE, WIDTH_KEY, int(width))
    _set(PROMPT_NODE, HEIGHT_KEY, int(height))
    _set(DURATION_NODE, DURATION_KEY, float(duration))
    _set(SEED_NODE, SEED_KEY, int(seed))

    # Ensure T2V path does not carry optional frame links from a polluted template
    prompt_inputs = out[PROMPT_NODE]["inputs"]
    prompt_inputs.pop(FIRST_FRAME_KEY, None)
    prompt_inputs.pop(LAST_FRAME_KEY, None)

    if first_image_name:
        out[FIRST_LOAD_NODE] = {
            "class_type": "LoadImage",
            "inputs": {"image": first_image_name.strip()},
        }
        prompt_inputs[FIRST_FRAME_KEY] = [FIRST_LOAD_NODE, 0]
    if last_image_name:
        out[LAST_LOAD_NODE] = {
            "class_type": "LoadImage",
            "inputs": {"image": last_image_name.strip()},
        }
        prompt_inputs[LAST_FRAME_KEY] = [LAST_LOAD_NODE, 0]

    return out
