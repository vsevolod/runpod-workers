"""ML-independent request parsing, compatibility aliases and common validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math

from PIL import Image

from .images import decode_image_string
from ..megapixels import resolve_megapixels


class RequestError(ValueError):
    """A validation error safe to return without restarting the worker."""


@dataclass(frozen=True)
class ModelAPI:
    name: str
    max_steps: int
    guidance_default: float
    samplers: tuple[str, ...]
    size_multiple: int
    max_side: int
    max_refs: int
    options: dict
    aliases: dict[str, str]  # Old top-level name -> model_options name
    standard_cfg: bool = False
    preserve_alpha: bool = False
    max_image_bytes: int | None = None
    max_source_side: int | None = None
    steps_by_variant: dict[str, int] = field(default_factory=dict)
    structured_prompt: bool = False

    def default_steps(self, options: dict) -> int:
        return self.steps_by_variant.get(options.get("variant"), 8)

    def common_schema(self) -> dict:
        side = lambda value: type(value) is int and 256 <= value <= self.max_side and value % self.size_multiple == 0
        return {
            "type": {"type": str, "required": False, "default": "image_generate", "constraints": lambda v: v in ("image_generate", "image_edit")},
            "prompt": {"type": dict if self.structured_prompt else str, "required": True},
            "negative_prompt": {"type": dict if self.structured_prompt else str, "required": False,
                                "default": None, "nullable": True},
            "width": {"type": int, "required": False, "default": 1024, "constraints": side},
            "height": {"type": int, "required": False, "default": 1024, "constraints": side},
            "seed": {"type": int, "required": False, "default": None, "nullable": True, "constraints": lambda v: -(2**63) <= v < 2**64},
            "num_inference_steps": {"type": int, "required": False, "default": 8, "constraints": lambda v: type(v) is int and 1 <= v <= self.max_steps},
            "guidance_scale": {"type": float, "required": False, "default": self.guidance_default, "constraints": lambda v: math.isfinite(v) and (not self.standard_cfg or v >= 0)},
            "sampler": {"type": str, "required": False, "default": "euler", "constraints": lambda v: v in self.samplers},
            "num_images": {"type": int, "required": False, "default": 1, "constraints": lambda v: type(v) is int and 1 <= v <= 4},
            "images": {"type": list, "required": False, "default": []},
            "reference_megapixels": {"type": float, "required": False, "default": "max", "special_values": ("max",),
                                     "constraints": lambda v: math.isfinite(v) and .001024 <= v <= 4.3008},
            "loras": {"type": list, "required": False, "default": [], "nullable": True},
        }

    def input_schema(self) -> dict:
        schema = self.common_schema()
        schema["model_options"] = {"type": dict, "required": False, "default": {}}
        for old, new in self.aliases.items():
            schema[old] = deepcopy(self.options[new])
        return schema


@dataclass(frozen=True)
class ParsedInput:
    values: dict
    raw_keys: set[str]


def _validate_field(name, value, rule):
    if isinstance(value, str) and value in rule.get("special_values", ()):
        return value
    if value is None and rule.get("nullable"):
        return None
    expected = rule["type"]
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RequestError(f"{name} must be a number")
        try:
            value = float(value)
        except OverflowError as err:
            raise RequestError(f"{name} must be finite") from err
    elif expected is int:
        if type(value) is not int:
            raise RequestError(f"{name} must be an integer")
    elif not isinstance(value, expected):
        raise RequestError(f"{name} must be a {expected.__name__}")
    if "constraints" in rule and not rule["constraints"](value):
        if name == "type":
            raise RequestError(f"unsupported type: {value!r}")
        raise RequestError(f"{name} does not meet the constraints for this model")
    return value


def parse_job_input(job, profile: ModelAPI) -> ParsedInput:
    if not isinstance(job, dict) or not isinstance(job.get("input"), dict):
        raise RequestError("Job must contain an 'input' object")
    raw = dict(job["input"])
    unknown = set(raw) - set(profile.input_schema())
    if unknown:
        raise RequestError(f"Unknown input field(s): {', '.join(sorted(unknown))}")
    options = raw.get("model_options", {})
    if not isinstance(options, dict):
        raise RequestError("model_options must be an object")
    options = dict(options)
    if "reference_megapixels" in options:
        budget_rule = profile.common_schema()["reference_megapixels"]
        nested_budget = _validate_field("model_options.reference_megapixels", options.pop("reference_megapixels"), budget_rule)
        if "reference_megapixels" in raw:
            top_budget = _validate_field("reference_megapixels", raw["reference_megapixels"], budget_rule)
            if top_budget != nested_budget:
                raise RequestError("Parameter conflict: reference_megapixels and model_options.reference_megapixels differ")
        raw["reference_megapixels"] = nested_budget
    unknown = set(options) - set(profile.options)
    if unknown:
        raise RequestError(f"Unknown model_options for {profile.name}: {', '.join(sorted(unknown))}")
    for old, new in profile.aliases.items():
        if old in raw:
            if new in options and options[new] != raw[old]:
                raise RequestError(f"Parameter conflict: {old} and model_options.{new} have different values")
            options[new] = raw[old]
    explicit_options = set(options)
    options = {
        name: _validate_field(f"model_options.{name}", options.get(name, deepcopy(rule.get("default"))), rule)
        for name, rule in profile.options.items()
    }
    values = {}
    for name, rule in profile.common_schema().items():
        if rule.get("required") and name not in raw:
            raise RequestError(f"{name} is required")
        value = raw.get(name, deepcopy(rule.get("default")))
        if name == "num_inference_steps" and name not in raw:
            value = profile.default_steps(options)
        values[name] = _validate_field(name, value, rule)
    values["model_options"] = options
    for old, new in profile.aliases.items():
        values[old] = options[new]
    # Expose only explicit legacy-equivalent keys: width/height and variant
    # defaults must not erase the client's intent after default injection.
    raw_keys = set(raw) - {"model_options"}
    raw_keys.update(old for old, new in profile.aliases.items() if new in explicit_options)
    return ParsedInput(values, raw_keys)


@dataclass(frozen=True)
class CommonRequest:
    type: str
    prompt: str | dict
    negative_prompt: str | dict | None
    seed: int | None
    num_inference_steps: int
    guidance_scale: float
    sampler: str
    num_images: int
    images: tuple[Image.Image, ...]
    reference_megapixels: float
    requested_reference_megapixels: float | str
    vram_gib: float | None


def normalize_common_fields(values: dict, profile: ModelAPI, *, raw_keys=None, default_steps=None,
                            megapixel_model=None) -> CommonRequest:
    schema = profile.common_schema()
    common = {}
    for name in ("type", "prompt", "negative_prompt", "seed", "num_inference_steps", "guidance_scale", "sampler", "num_images", "images"):
        value = values.get(name, deepcopy(schema[name].get("default")))
        if name == "num_inference_steps" and default_steps is not None and name not in (raw_keys or ()):
            value = default_steps
        common[name] = _validate_field(name, value, schema[name])
    if profile.structured_prompt:
        if not common["prompt"]:
            raise RequestError("prompt must be a non-empty JSON object")
        common["prompt"] = deepcopy(common["prompt"])
        common["negative_prompt"] = deepcopy(common["negative_prompt"])
    else:
        if not common["prompt"].strip():
            raise RequestError("prompt must be a non-empty string")
        common["prompt"] = common["prompt"].strip()
    images = common["images"]
    if common["type"] == "image_generate":
        if images:
            raise RequestError("images must be empty for image_generate")
    else:
        if not 1 <= len(images) <= profile.max_refs:
            count = "1 or 2" if profile.max_refs == 2 else f"1 to {profile.max_refs}"
            raise RequestError(f"image_edit requires {count} entries in images[]")
        if common["num_images"] != 1:
            raise RequestError("image_edit requires num_images == 1")
    common["images"] = tuple(decode_image_string(
        image, preserve_alpha=profile.preserve_alpha,
        max_bytes=profile.max_image_bytes, max_side=profile.max_source_side,
    ) for image in images)
    requested = _validate_field("reference_megapixels", values.get("reference_megapixels", "max"),
                                schema["reference_megapixels"])
    budget = resolve_megapixels(requested, megapixel_model or profile.name)
    resolved = _validate_field("resolved_reference_megapixels", budget.megapixels, schema["reference_megapixels"])
    common.update(reference_megapixels=resolved, requested_reference_megapixels=requested,
                  vram_gib=budget.vram_gib)
    if profile.standard_cfg and not profile.structured_prompt:
        negative = common["negative_prompt"]
        negative = negative.strip() or None if negative is not None else None
        if common["guidance_scale"] > 1 and negative is None:
            negative = ""
        common["negative_prompt"] = negative
    return CommonRequest(**common)
