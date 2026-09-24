"""RunPod input validation schema for Qwen-Image-2.1."""


def _canvas_side(value: int) -> bool:
    return isinstance(value, int) and 256 <= value <= 2752 and value % 32 == 0


def _optional_resolution(value) -> bool:
    return value is None or _canvas_side(value)


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
        "constraints": _canvas_side,
    },
    "height": {
        "type": int,
        "required": False,
        "default": 1024,
        "constraints": _canvas_side,
    },
    "seed": {
        "type": int,
        "required": False,
        "default": None,
    },
    "num_inference_steps": {
        "type": int,
        "required": False,
        "default": 40,
        "constraints": lambda steps: isinstance(steps, int) and 1 <= steps <= 80,
    },
    # Maps to Diffusers true_cfg_scale. Values <= 1 leave classifier-free guidance off.
    "guidance_scale": {
        "type": float,
        "required": False,
        "default": 1.0,
    },
    "num_images": {
        "type": int,
        "required": False,
        "default": 1,
        "constraints": lambda count: isinstance(count, int) and 1 <= count <= 4,
    },
    "images": {
        "type": list,
        "required": False,
        "default": [],
    },
    # Condition-image area, and the edit canvas when width/height are omitted.
    # None lets the handler derive it from the canvas (or 1024 for source-sized edits).
    "output_resolution": {
        "type": int,
        "required": False,
        "default": None,
        "constraints": _optional_resolution,
    },
}
