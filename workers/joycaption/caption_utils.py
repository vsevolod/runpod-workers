"""Pure helpers for JoyCaption worker (unit-testable without GPU)."""

from __future__ import annotations

import base64
import binascii
import os
import re
from io import BytesIO
from typing import Mapping

from PIL import Image, UnidentifiedImageError

DEFAULT_MODEL_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"
DEFAULT_PROMPT = (
    "Write a long descriptive caption for this image in a formal tone."
)
SYSTEM_MESSAGE = "You are a helpful image captioner."
DEFAULT_MAX_IMAGE_PIXELS = 25_000_000

_DATA_URL_RE = re.compile(
    r"^data:image/(?P<fmt>[a-zA-Z0-9.+-]+);base64,(?P<data>.+)$",
    re.DOTALL,
)


class ImageError(ValueError):
    """Invalid image payload or pixel limit exceeded."""


def parse_local_files_only(
    env: Mapping[str, str] | None = None,
) -> bool:
    """Parse LOCAL_FILES_ONLY for explicit from_pretrained(local_files_only=...)."""
    source = env if env is not None else os.environ
    return str(source.get("LOCAL_FILES_ONLY", "")).lower() in {"1", "true", "yes"}


def parse_max_image_pixels(
    env: Mapping[str, str] | None = None,
) -> int:
    """Max width*height after decode; default 25_000_000 (~5000x5000)."""
    source = env if env is not None else os.environ
    raw = source.get("MAX_IMAGE_PIXELS")
    if raw is None or str(raw).strip() == "":
        return DEFAULT_MAX_IMAGE_PIXELS
    try:
        value = int(raw)
    except (TypeError, ValueError) as err:
        raise ValueError(f"invalid MAX_IMAGE_PIXELS: {raw!r}") from err
    if value < 1:
        raise ValueError(f"MAX_IMAGE_PIXELS must be >= 1, got {value}")
    return value


def resolve_prompt(prompt: str | None) -> str:
    """Return override prompt if non-empty, else default descriptive EN prompt."""
    if prompt is None:
        return DEFAULT_PROMPT
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be str or None, got {type(prompt).__name__}")
    stripped = prompt.strip()
    if not stripped:
        return DEFAULT_PROMPT
    return stripped


def _strip_data_url(image_b64: str) -> str:
    text = image_b64.strip()
    match = _DATA_URL_RE.match(text)
    if match:
        return match.group("data")
    # Tolerate whitespace/newlines in raw base64
    if text.startswith("data:"):
        # Malformed data URL
        raise ImageError("invalid image: malformed data URL (expected data:image/...;base64,...)")
    return "".join(text.split())


def decode_image_base64(
    image_b64: str,
    *,
    max_pixels: int | None = None,
) -> Image.Image:
    """Decode raw base64 or data URL to RGB PIL Image; enforce pixel limit."""
    if not isinstance(image_b64, str) or not image_b64.strip():
        raise ImageError("invalid image: empty or missing base64 payload")

    if max_pixels is None:
        max_pixels = parse_max_image_pixels()

    # Align Pillow bomb guard with our limit (Pillow default is 178M+)
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels

    try:
        payload = _strip_data_url(image_b64)
        try:
            raw = base64.b64decode(payload, validate=False)
        except (binascii.Error, ValueError) as err:
            raise ImageError(f"invalid image: base64 decode failed: {err}") from err
        if not raw:
            raise ImageError("invalid image: empty binary after base64 decode")

        try:
            image = Image.open(BytesIO(raw))
            image.load()
        except Image.DecompressionBombError as err:
            raise ImageError(
                f"image too large: exceeds MAX_IMAGE_PIXELS={max_pixels}"
            ) from err
        except UnidentifiedImageError as err:
            raise ImageError("invalid image: not a recognized image format") from err
        except OSError as err:
            raise ImageError(f"invalid image: {err}") from err

        width, height = image.size
        pixels = width * height
        if pixels > max_pixels:
            raise ImageError(
                f"image too large: {width}x{height} ({pixels} px) exceeds "
                f"MAX_IMAGE_PIXELS={max_pixels}"
            )

        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    finally:
        Image.MAX_IMAGE_PIXELS = previous
