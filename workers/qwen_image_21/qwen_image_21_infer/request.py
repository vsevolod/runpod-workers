"""Request contract for Qwen-Image-2.1 image_generate / image_edit jobs."""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

MAX_REFS = 10
MIN_SIDE = 256
MAX_SIDE = 2752
# Largest official preset is 4:3 (2400×1792). Anything bigger is rejected.
MAX_AREA = 2400 * 1792
MULTIPLE = 32
DEFAULT_OUTPUT_RESOLUTION = 1024
DEFAULT_STEPS = 40
DEFAULT_GUIDANCE = 1.0
MAX_STEPS = 80
MAX_IMAGES = 4
MAX_SOURCE_SIDE = 8192
MAX_DECODED_BYTES = 25 * 1024 * 1024

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


class RequestError(ValueError):
    """Safe client-facing validation error."""


@dataclass(frozen=True)
class NormalizedRequest:
    type: str  # "image_generate" | "image_edit"
    prompt: str
    negative_prompt: str | None
    width: int | None  # None when the edit canvas follows the last reference
    height: int | None
    size_from_source: bool
    seed: int | None
    num_inference_steps: int
    guidance_scale: float  # passed through as true_cfg_scale; <= 1 disables CFG
    num_images: int
    images: tuple[Image.Image, ...]  # 0 for generate; 1..10 for edit
    output_resolution: int


def decode_image_string(value: str) -> Image.Image:
    """Accept a data URL or raw base64. Keep an alpha channel when the file has one."""
    if not isinstance(value, str) or not value.strip():
        raise RequestError("each images[] entry must be a non-empty base64 string")
    payload = value.strip()
    if payload.startswith("data:"):
        try:
            header, payload = payload.split(",", 1)
        except ValueError as err:
            raise RequestError("invalid data URL in images[]") from err
        if ";base64" not in header:
            raise RequestError("images[] data URL must be base64")
    try:
        raw = base64.b64decode(payload, validate=False)
    except Exception as err:
        raise RequestError("images[] is not valid base64") from err
    if not raw:
        raise RequestError("images[] decoded to empty bytes")
    if len(raw) > MAX_DECODED_BYTES:
        raise RequestError(
            f"images[] entry is {len(raw)} bytes; limit is {MAX_DECODED_BYTES}"
        )
    try:
        image = Image.open(BytesIO(raw))
        image.load()
    except Exception as err:
        raise RequestError("images[] could not be decoded as an image") from err
    width, height = image.size
    if width > MAX_SOURCE_SIDE or height > MAX_SOURCE_SIDE:
        raise RequestError(
            f"images[] side must be <= {MAX_SOURCE_SIDE}px (got {width}x{height})"
        )
    if image.mode == "RGBA":
        return image
    if image.mode in ("LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        return image.convert("RGBA")
    return image.convert("RGB")


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


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RequestError(f"{field} must be a string")
    text = value.strip()
    return text or None


def _derived_output_resolution(width: int, height: int) -> int:
    side = int(round(math.sqrt(width * height)))
    return max(MIN_SIDE, min(MAX_SIDE, side))


def normalize_job_input(
    validated: dict,
    *,
    raw_keys: set[str] | None = None,
) -> NormalizedRequest:
    """Turn a schema-validated job input into a pipeline request.

    ``raw_keys`` is ``set(job['input'].keys())`` before RunPod fills defaults.
    Edit uses it so an omitted width/height follows the last reference instead
    of the schema default canvas.
    """
    raw_keys = set(raw_keys or ())

    req_type = validated.get("type") or "image_generate"
    if req_type not in ("image_generate", "image_edit"):
        raise RequestError(f"unsupported type: {req_type!r}")

    prompt = validated.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise RequestError("prompt must be a non-empty string")

    images_raw = validated.get("images") or []
    if not isinstance(images_raw, list):
        raise RequestError("images must be a list")

    steps = int(validated.get("num_inference_steps") or DEFAULT_STEPS)
    if not 1 <= steps <= MAX_STEPS:
        raise RequestError(f"num_inference_steps must be in 1..{MAX_STEPS}")

    guidance_raw = validated.get("guidance_scale")
    guidance = DEFAULT_GUIDANCE if guidance_raw is None else float(guidance_raw)
    if not math.isfinite(guidance) or guidance < 0:
        raise RequestError("guidance_scale must be a finite number >= 0")

    negative = _optional_text(validated.get("negative_prompt"), "negative_prompt")
    if guidance > 1 and negative is None:
        raise RequestError(
            "guidance_scale > 1 requires a non-empty negative_prompt "
            "(true CFG is off at guidance_scale <= 1)"
        )

    num_images = int(validated.get("num_images") or 1)
    if not 1 <= num_images <= MAX_IMAGES:
        raise RequestError(f"num_images must be in 1..{MAX_IMAGES}")

    output_resolution = validated.get("output_resolution")
    if output_resolution is not None:
        output_resolution = int(output_resolution)
        if not MIN_SIDE <= output_resolution <= MAX_SIDE:
            raise RequestError(
                f"output_resolution must be in {MIN_SIDE}..{MAX_SIDE} "
                f"(got {output_resolution})"
            )

    if req_type == "image_generate":
        if images_raw:
            raise RequestError("images must be empty for image_generate")
        images: tuple[Image.Image, ...] = ()
        size_from_source = False
        width = int(validated["width"])
        height = int(validated["height"])
        _check_canvas(width, height)
        if output_resolution is None:
            output_resolution = _derived_output_resolution(width, height)
    else:
        count = len(images_raw)
        if count < 1 or count > MAX_REFS:
            raise RequestError(
                f"image_edit requires 1 to {MAX_REFS} entries in images[]"
            )
        if num_images != 1:
            raise RequestError("image_edit requires num_images == 1")
        images = tuple(decode_image_string(item) for item in images_raw)
        has_width = "width" in raw_keys
        has_height = "height" in raw_keys
        if has_width ^ has_height:
            raise RequestError(
                "pass both width and height, or neither "
                "(size follows the last reference)"
            )
        size_from_source = not has_width and not has_height
        if size_from_source:
            width, height = None, None
            if output_resolution is None:
                output_resolution = DEFAULT_OUTPUT_RESOLUTION
        else:
            width = int(validated["width"])
            height = int(validated["height"])
            _check_canvas(width, height)
            if output_resolution is None:
                output_resolution = _derived_output_resolution(width, height)

    seed = validated.get("seed")
    if seed is not None:
        seed = int(seed)

    return NormalizedRequest(
        type=req_type,
        prompt=str(prompt).strip(),
        negative_prompt=negative,
        width=width,
        height=height,
        size_from_source=size_from_source,
        seed=seed,
        num_inference_steps=steps,
        guidance_scale=guidance,
        num_images=num_images,
        images=images,
        output_resolution=output_resolution,
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
    if norm.negative_prompt:
        kwargs["negative_prompt"] = norm.negative_prompt
    if norm.width is not None and norm.height is not None:
        kwargs["width"] = norm.width
        kwargs["height"] = norm.height
    if norm.images:
        kwargs["image"] = list(norm.images)
    return kwargs
