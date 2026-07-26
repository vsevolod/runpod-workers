"""Unit tests for joycaption helpers (no GPU / no model weights)."""

from __future__ import annotations

import base64
from io import BytesIO

import pytest
from PIL import Image

from caption_utils import (
    DEFAULT_MAX_IMAGE_PIXELS,
    DEFAULT_PROMPT,
    ImageError,
    decode_image_base64,
    parse_local_files_only,
    parse_max_image_pixels,
    resolve_prompt,
)


def _png_b64(width: int = 8, height: int = 8, color=(10, 20, 30)) -> str:
    img = Image.new("RGB", (width, height), color=color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_parse_local_files_only_truthy():
    assert parse_local_files_only({"LOCAL_FILES_ONLY": "1"}) is True
    assert parse_local_files_only({"LOCAL_FILES_ONLY": "true"}) is True
    assert parse_local_files_only({"LOCAL_FILES_ONLY": "YES"}) is True


def test_parse_local_files_only_falsy():
    assert parse_local_files_only({}) is False
    assert parse_local_files_only({"LOCAL_FILES_ONLY": ""}) is False
    assert parse_local_files_only({"LOCAL_FILES_ONLY": "0"}) is False
    assert parse_local_files_only({"LOCAL_FILES_ONLY": "false"}) is False


def test_parse_max_image_pixels_default():
    assert parse_max_image_pixels({}) == DEFAULT_MAX_IMAGE_PIXELS


def test_parse_max_image_pixels_custom():
    assert parse_max_image_pixels({"MAX_IMAGE_PIXELS": "1000"}) == 1000


def test_parse_max_image_pixels_invalid():
    with pytest.raises(ValueError, match="invalid MAX_IMAGE_PIXELS"):
        parse_max_image_pixels({"MAX_IMAGE_PIXELS": "nope"})
    with pytest.raises(ValueError, match="must be >= 1"):
        parse_max_image_pixels({"MAX_IMAGE_PIXELS": "0"})


def test_resolve_prompt_default_when_missing_or_blank():
    assert resolve_prompt(None) == DEFAULT_PROMPT
    assert resolve_prompt("") == DEFAULT_PROMPT
    assert resolve_prompt("   ") == DEFAULT_PROMPT


def test_resolve_prompt_override():
    assert resolve_prompt("Describe this briefly.") == "Describe this briefly."
    assert resolve_prompt("  x  ") == "x"


def test_decode_raw_base64():
    b64 = _png_b64(4, 5)
    img = decode_image_base64(b64, max_pixels=100)
    assert img.size == (4, 5)
    assert img.mode == "RGB"


def test_decode_data_url():
    b64 = _png_b64(3, 3)
    data_url = f"data:image/png;base64,{b64}"
    img = decode_image_base64(data_url, max_pixels=100)
    assert img.size == (3, 3)


def test_decode_empty_raises():
    with pytest.raises(ImageError, match="empty"):
        decode_image_base64("", max_pixels=100)
    with pytest.raises(ImageError, match="empty"):
        decode_image_base64("   ", max_pixels=100)


def test_decode_not_image_raises():
    junk = base64.b64encode(b"not-an-image").decode("ascii")
    with pytest.raises(ImageError, match="not a recognized"):
        decode_image_base64(junk, max_pixels=100)


def test_decode_pixel_limit_raises():
    b64 = _png_b64(10, 10)
    with pytest.raises(ImageError, match="image too large"):
        decode_image_base64(b64, max_pixels=50)
