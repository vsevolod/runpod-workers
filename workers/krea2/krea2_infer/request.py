"""Request contract for image_generate / image_edit job inputs."""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from io import BytesIO

from PIL import Image


class RequestError(ValueError):
    """Safe client-facing validation error."""


@dataclass(frozen=True)
class NormalizedRequest:
    type: str  # "image_generate" | "image_edit"
    prompt: str
    negative_prompt: str | None
    width: int | None  # None when size_from_source
    height: int | None
    size_from_source: bool
    seed: int | None
    num_inference_steps: int
    guidance_scale: float
    mu: float | None
    num_images: int
    images: tuple[Image.Image, ...]  # 0 for generate; 1 for edit
    grounding_px: int
    ref_boost: float
    fit_mode: str  # "fit" | "crop"
    # loras left to handler / LoRAManager


def decode_image_string(s: str) -> Image.Image:
    """Accept data:image/...;base64,... or raw base64; return RGB PIL.Image."""
    if not isinstance(s, str) or not s.strip():
        raise RequestError("each images[] entry must be a non-empty base64 string")
    payload = s.strip()
    if payload.startswith("data:"):
        # data:image/png;base64,<payload>
        try:
            header, b64 = payload.split(",", 1)
        except ValueError as err:
            raise RequestError("invalid data URL in images[]") from err
        if ";base64" not in header:
            raise RequestError("images[] data URL must be base64")
        payload = b64
    try:
        raw = base64.b64decode(payload, validate=False)
    except Exception as err:
        raise RequestError("images[] is not valid base64") from err
    try:
        img = Image.open(BytesIO(raw))
        img.load()
    except Exception as err:
        raise RequestError("images[] could not be decoded as an image") from err
    return img.convert("RGB")


def normalize_job_input(
    validated: dict,
    *,
    raw_keys: set[str] | None = None,
) -> NormalizedRequest:
    """
    validated: output of runpod validate() (defaults applied).
    raw_keys: set(job['input'].keys()) BEFORE validate — used only for edit size.
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

    if req_type == "image_generate":
        if images_raw:
            raise RequestError("images must be empty for image_generate")
        images: tuple[Image.Image, ...] = ()
        size_from_source = False
        width = int(validated["width"])
        height = int(validated["height"])
        if width % 16 != 0 or height % 16 != 0:
            raise RequestError(
                f"width and height must be multiples of 16 (got {width}x{height})"
            )
    else:
        if len(images_raw) != 1:
            raise RequestError("image_edit requires exactly one entry in images[]")
        images = (decode_image_string(images_raw[0]),)
        num_images = int(validated.get("num_images") or 1)
        if num_images != 1:
            raise RequestError("image_edit requires num_images == 1")
        size_from_source = ("width" not in raw_keys) and ("height" not in raw_keys)
        if size_from_source:
            width, height = None, None
        else:
            width = int(validated["width"])
            height = int(validated["height"])
            if width % 16 != 0 or height % 16 != 0:
                raise RequestError(
                    f"width and height must be multiples of 16 (got {width}x{height})"
                )

    fit_mode = validated.get("fit_mode") or "fit"
    if fit_mode not in ("fit", "crop"):
        raise RequestError("fit_mode must be 'fit' or 'crop'")

    grounding_px = int(
        validated.get("grounding_px") if validated.get("grounding_px") is not None else 768
    )
    if grounding_px != 0 and not (256 <= grounding_px <= 2048):
        raise RequestError("grounding_px must be 0 or in 256..2048")

    ref_boost = float(
        validated.get("ref_boost") if validated.get("ref_boost") is not None else 1.0
    )
    if not math.isfinite(ref_boost) or ref_boost < 0:
        raise RequestError("ref_boost must be a finite number >= 0")

    return NormalizedRequest(
        type=req_type,
        prompt=str(prompt).strip(),
        negative_prompt=validated.get("negative_prompt"),
        width=width,
        height=height,
        size_from_source=size_from_source,
        seed=validated.get("seed"),
        num_inference_steps=int(validated.get("num_inference_steps") or 8),
        guidance_scale=float(validated.get("guidance_scale") or 0.0),
        mu=validated.get("mu"),
        num_images=int(validated.get("num_images") or 1),
        images=images,
        grounding_px=grounding_px,
        ref_boost=ref_boost,
        fit_mode=fit_mode,
    )
