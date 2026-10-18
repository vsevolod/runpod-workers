"""Worker capabilities and model-specific options; no model runtime imports."""

import math

from .contract import ModelAPI


QWEN_API = ModelAPI(
    name="qwen_image_21", max_steps=200, guidance_default=1.,
    samplers=("euler", "heun", "er_sde"), size_multiple=32, max_side=2752, max_refs=10,
    standard_cfg=True, preserve_alpha=True, max_image_bytes=25 * 1024 * 1024, max_source_side=8192,
    steps_by_variant={"base": 40, "turbo": 8},
    options={
        "variant": {"type": str, "required": False, "default": "turbo", "constraints": lambda v: v in ("base", "turbo")},
        "output_resolution": {"type": int, "required": False, "default": None, "nullable": True,
                              "constraints": lambda v: v is None or type(v) is int and 256 <= v <= 2752 and v % 32 == 0},
        "reference_megapixels": {"type": float, "required": False, "default": 1.,
                                 "constraints": lambda v: math.isfinite(v) and .001024 <= v <= 4.3008},
    },
    aliases={"model_variant": "variant", "output_resolution": "output_resolution"},
)

KREA_API = ModelAPI(
    name="krea2", max_steps=64, guidance_default=0.,
    samplers=("euler",), size_multiple=16, max_side=2048, max_refs=2,
    options={
        "mu": {"type": float, "required": False, "default": 1.15, "nullable": True, "constraints": math.isfinite},
        "grounding_px": {"type": int, "required": False, "default": 768, "constraints": lambda v: v == 0 or 256 <= v <= 2048},
        "ref_boost": {"type": float, "required": False, "default": 1., "constraints": lambda v: math.isfinite(v) and v >= 0},
        "ref_boost_a": {"type": float, "required": False, "default": 1., "constraints": lambda v: math.isfinite(v) and v >= 0},
        "ref_position": {"type": str, "required": False, "default": "aligned", "constraints": lambda v: v in ("aligned", "beside", "bag_of_features")},
        "ref_position_a": {"type": str, "required": False, "default": "aligned", "constraints": lambda v: v in ("aligned", "beside", "bag_of_features")},
        "fit_mode": {"type": str, "required": False, "default": "fit", "constraints": lambda v: v in ("fit", "crop")},
    },
    aliases={name: name for name in ("mu", "grounding_px", "ref_boost", "ref_boost_a", "ref_position", "ref_position_a", "fit_mode")},
)
