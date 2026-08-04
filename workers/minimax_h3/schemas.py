"""RunPod INPUT_SCHEMA for MiniMax H3 T2V."""

from __future__ import annotations

INPUT_SCHEMA = {
    "prompt": {"type": str, "required": True},
    "width": {
        "type": int,
        "required": False,
        "default": 864,
        "constraints": lambda n: isinstance(n, int)
        and not isinstance(n, bool)
        and n >= 32
        and n % 32 == 0,
    },
    "height": {
        "type": int,
        "required": False,
        "default": 480,
        "constraints": lambda n: isinstance(n, int)
        and not isinstance(n, bool)
        and n >= 32
        and n % 32 == 0,
    },
    "duration": {
        "type": float,
        "required": False,
        "default": 5.0,
        "constraints": lambda d: isinstance(d, (int, float))
        and not isinstance(d, bool)
        and 5.0 <= float(d) <= 14.375,
    },
    "seed": {
        "type": int,
        "required": False,
        "default": -1,  # sentinel → random in normalize
        "constraints": lambda s: isinstance(s, int)
        and not isinstance(s, bool)
        and s >= -1,
    },
}
