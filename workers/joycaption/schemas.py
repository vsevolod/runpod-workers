"""RunPod input validation schema for JoyCaption worker."""

INPUT_SCHEMA = {
    "image": {
        "type": str,
        "required": True,
    },
    "prompt": {
        "type": str,
        "required": False,
        "default": None,
    },
    "max_new_tokens": {
        "type": int,
        "required": False,
        "default": 512,
        "constraints": lambda n: isinstance(n, int) and 1 <= n <= 1024,
    },
    "temperature": {
        "type": float,
        "required": False,
        "default": 0.6,
        "constraints": lambda t: isinstance(t, (int, float)) and 0.0 <= float(t) <= 2.0,
    },
    "top_p": {
        "type": float,
        "required": False,
        "default": 0.9,
        "constraints": lambda p: isinstance(p, (int, float)) and 0.0 < float(p) <= 1.0,
    },
}
