"""Request contract for Qwen-Image-2.1 image_generate / image_edit jobs."""

from __future__ import annotations

import math
from dataclasses import dataclass
import sys
from pathlib import Path

from PIL import Image

from qwen_image_21_infer.runtime_config import BASE_STEPS, TURBO_STEPS, resolve_model_variant
from qwen_image_21_infer.sampling import comfy_simple_sigmas

workers_dir = Path(__file__).resolve().parents[2]
if (workers_dir / "shared").is_dir():
    sys.path.insert(0, str(workers_dir))
from shared.api import CommonRequest, QWEN_API, RequestError, normalize_common_fields
from shared.api.images import decode_image_string as _decode_image

MIN_SIDE = 256
MAX_SIDE = QWEN_API.max_side
# Largest official preset is 4:3 (2400×1792). Anything bigger is rejected.
MAX_AREA = 2400 * 1792
MULTIPLE = QWEN_API.size_multiple
MAX_SOURCE_SIDE = QWEN_API.max_source_side
MAX_DECODED_BYTES = QWEN_API.max_image_bytes

# Sizes from the Qwen-Image-2.1 model card. All are multiples of 32.
OFFICIAL_PRESETS: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}


@dataclass(frozen=True)
class NormalizedRequest(CommonRequest):
    model_variant: str
    width: int
    height: int
    size_from_source: bool
    output_resolution: int
    requested_output_resolution: int | None

    @property
    def scheduler(self) -> str:
        return "checkpoint" if self.model_variant == "turbo" and self.num_inference_steps == TURBO_STEPS else "simple"


def decode_image_string(value: str) -> Image.Image:
    return _decode_image(value, preserve_alpha=True, max_bytes=MAX_DECODED_BYTES, max_side=MAX_SOURCE_SIDE)


def _check_canvas(width: int, height: int) -> None:
    if width % MULTIPLE != 0 or height % MULTIPLE != 0:
        raise RequestError(
            f"width and height must be multiples of {MULTIPLE} (got {width}x{height})"
        )
    if not (MIN_SIDE <= width <= MAX_SIDE and MIN_SIDE <= height <= MAX_SIDE):
        raise RequestError(
            f"width and height must be in {MIN_SIDE}..{MAX_SIDE} (got {width}x{height})"
        )
    if width * height > MAX_AREA:
        raise RequestError(
            f"width*height must be <= {MAX_AREA} (got {width}x{height}). "
            "Official presets fit, including 2048x2048, 2400x1792, 2528x1696, "
            "2752x1536 and their swaps."
        )


def _derived_output_resolution(width: int, height: int) -> int:
    side = int(round(math.sqrt(width * height)))
    return max(MIN_SIDE, min(MAX_SIDE, side))


def _native_size(image: Image.Image) -> tuple[int, int]:
    return tuple(max(MULTIPLE, round(side / MULTIPLE) * MULTIPLE) for side in image.size)


def _canvas_at_resolution(image: Image.Image, resolution: int) -> tuple[int, int]:
    ratio = image.width / image.height
    return (
        round(math.sqrt(resolution * resolution * ratio) / MULTIPLE) * MULTIPLE,
        round(math.sqrt(resolution * resolution / ratio) / MULTIPLE) * MULTIPLE,
    )


def normalize_job_input(
    validated: dict,
    *,
    raw_keys: set[str] | None = None,
) -> NormalizedRequest:
    """Apply Qwen canvas/reference geometry to a parsed shared API request.

    ``raw_keys`` contains explicit keys before the shared parser fills defaults,
    including the legacy equivalents of explicitly nested model options.
    Edit uses it so an omitted width/height follows the last reference instead
    of the schema default canvas. Reference images keep their native sizes.
    """
    raw_keys = set(raw_keys or ())

    try:
        variant = resolve_model_variant(validated.get("model_variant", "turbo"))
    except (ValueError, TypeError) as err:
        raise RequestError("model_variant must be 'base' or 'turbo'") from err
    default_steps = TURBO_STEPS if variant == "turbo" else BASE_STEPS
    common = normalize_common_fields(validated, QWEN_API, raw_keys=raw_keys, default_steps=default_steps)
    images = common.images

    output_resolution = validated.get("output_resolution")
    if output_resolution is not None:
        output_resolution = int(output_resolution)
        if not MIN_SIDE <= output_resolution <= MAX_SIDE:
            raise RequestError(
                f"output_resolution must be in {MIN_SIDE}..{MAX_SIDE} "
                f"(got {output_resolution})"
            )

    if common.type == "image_generate":
        size_from_source = False
        width = int(validated["width"])
        height = int(validated["height"])
        _check_canvas(width, height)
        if output_resolution is None:
            output_resolution = _derived_output_resolution(width, height)
    else:
        for index, image in enumerate(images, 1):
            native_w, native_h = _native_size(image)
            if native_w > MAX_SIDE or native_h > MAX_SIDE or native_w * native_h > MAX_AREA:
                raise RequestError(
                    f"reference {index} native size {native_w}x{native_h} exceeds "
                    f"the supported limit ({MAX_SIDE}px per side, {MAX_AREA} pixels); "
                    "resize the reference before submitting"
                )
        has_width = "width" in raw_keys
        has_height = "height" in raw_keys
        if has_width ^ has_height:
            raise RequestError(
                "pass both width and height, or neither "
                "(size follows the last reference)"
            )
        size_from_source = not has_width and not has_height
        if size_from_source:
            if output_resolution is None:
                width, height = _native_size(images[-1])
                output_resolution = _derived_output_resolution(width, height)
            else:
                width, height = _canvas_at_resolution(images[-1], output_resolution)
            _check_canvas(width, height)
        else:
            width = int(validated["width"])
            height = int(validated["height"])
            _check_canvas(width, height)
            if output_resolution is None:
                output_resolution = _derived_output_resolution(width, height)

    return NormalizedRequest(
        **vars(common), model_variant=variant,
        width=width, height=height, size_from_source=size_from_source,
        output_resolution=output_resolution,
        requested_output_resolution=validated.get("output_resolution"),
    )


def build_pipe_kwargs(norm: NormalizedRequest) -> dict:
    """Arguments for ``QwenImage21Pipeline.__call__``, without the RNG.

    ``use_kv_cache`` is pinned on. Diffusers notes that flipping it changes
    the sample in reduced precision, so a seed stays comparable only while
    the flag stays fixed.
    """
    kwargs: dict = {
        "prompt": norm.prompt,
        "num_inference_steps": norm.num_inference_steps,
        "num_images_per_prompt": norm.num_images,
        "true_cfg_scale": norm.guidance_scale,
        "output_resolution": norm.output_resolution,
        "use_kv_cache": True,
    }
    if norm.scheduler == "simple":
        kwargs["sigmas"] = comfy_simple_sigmas(norm.num_inference_steps)
    # Eight-step Turbo keeps its saved grid; explicit other counts override it.
    if norm.negative_prompt is not None:
        kwargs["negative_prompt"] = norm.negative_prompt
    if norm.width is not None and norm.height is not None:
        kwargs["width"] = norm.width
        kwargs["height"] = norm.height
    if norm.images:
        kwargs["image"] = list(norm.images)
        kwargs["reference_resolution"] = 0
    return kwargs
