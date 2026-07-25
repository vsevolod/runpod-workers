"""RunPod input validation schema for Krea 2 Turbo."""

INPUT_SCHEMA = {
    "type": {
        "type": str,
        "required": False,
        "default": "image_generate",
    },
    "prompt": {
        "type": str,
        "required": True,
    },
    "negative_prompt": {
        "type": str,
        "required": False,
        "default": None,
    },
    "width": {
        "type": int,
        "required": False,
        "default": 1024,
        "constraints": lambda w: isinstance(w, int) and 256 <= w <= 2048,
    },
    "height": {
        "type": int,
        "required": False,
        "default": 1024,
        "constraints": lambda h: isinstance(h, int) and 256 <= h <= 2048,
    },
    "seed": {
        "type": int,
        "required": False,
        "default": None,
    },
    "num_inference_steps": {
        "type": int,
        "required": False,
        "default": 8,
        "constraints": lambda s: isinstance(s, int) and 1 <= s <= 64,
    },
    "guidance_scale": {
        "type": float,
        "required": False,
        "default": 0.0,
    },
    "mu": {
        "type": float,
        "required": False,
        "default": 1.15,
    },
    "num_images": {
        "type": int,
        "required": False,
        "default": 1,
        "constraints": lambda n: isinstance(n, int) and 1 <= n <= 4,
    },
    "images": {
        "type": list,
        "required": False,
        "default": [],
    },
    "grounding_px": {
        "type": int,
        "required": False,
        "default": 768,
    },
    "ref_boost": {
        "type": float,
        "required": False,
        "default": 1.0,
    },
    "fit_mode": {
        "type": str,
        "required": False,
        "default": "fit",
    },
    "loras": {
        "type": list,
        "required": False,
        "default": [],
    },
}
