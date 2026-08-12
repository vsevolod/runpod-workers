"""Resolve product first/last image refs (URL or base64) and stage for Comfy LoadImage."""

from __future__ import annotations

import base64
import binascii
import re
from io import BytesIO
from pathlib import Path
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError

# Defaults match design spec (override via kwargs in callers if needed)
DEFAULT_MAX_BYTES: Final[int] = 20 * 1024 * 1024
DEFAULT_MAX_PIXELS: Final[int] = 4096 * 4096
URL_TIMEOUT_S: Final[float] = 60.0

_DATA_URL_RE = re.compile(
    r"^data:image/(?P<fmt>png|jpeg|jpg|webp);base64,(?P<data>.+)$",
    re.IGNORECASE | re.DOTALL,
)

# Magic → preferred extension for staged file
_MAGIC: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"RIFF", "webp"),  # need WEBP at offset 8; refined below
]


class ImageInputError(ValueError):
    """Invalid product image (user-facing message)."""


def _strip_data_url(text: str) -> bytes:
    text = text.strip()
    match = _DATA_URL_RE.match(text)
    if match:
        payload = match.group("data")
    elif text.startswith("data:"):
        raise ImageInputError(
            "invalid image: malformed data URL "
            "(expected data:image/png|jpeg|jpg|webp;base64,...)"
        )
    else:
        payload = "".join(text.split())
    try:
        raw = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as err:
        raise ImageInputError(f"invalid image: base64 decode failed: {err}") from err
    if not raw:
        raise ImageInputError("invalid image: empty binary after base64 decode")
    return raw


def _fetch_url(url: str, *, max_bytes: int, timeout_s: float) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ImageInputError(
            f"invalid image URL scheme {parsed.scheme!r}; only http/https allowed"
        )
    req = Request(url, headers={"User-Agent": "minimax-h3-comfy-worker/1.0"})
    try:
        with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 — scheme checked
            # Prefer Content-Length guard when present
            cl = resp.headers.get("Content-Length")
            if cl is not None:
                try:
                    if int(cl) > max_bytes:
                        raise ImageInputError(
                            f"image too large: Content-Length {cl} > {max_bytes} bytes"
                        )
                except ValueError:
                    pass
            chunks: list[bytes] = []
            total = 0
            while True:
                block = resp.read(1024 * 256)
                if not block:
                    break
                total += len(block)
                if total > max_bytes:
                    raise ImageInputError(
                        f"image too large: exceeds {max_bytes} bytes while downloading"
                    )
                chunks.append(block)
    except ImageInputError:
        raise
    except HTTPError as err:
        raise ImageInputError(f"image URL HTTP {err.code}: {err.reason}") from err
    except URLError as err:
        raise ImageInputError(f"image URL fetch failed: {err.reason}") from err
    except TimeoutError as err:
        raise ImageInputError("image URL fetch timed out") from err
    raw = b"".join(chunks)
    if not raw:
        raise ImageInputError("invalid image: empty response body")
    return raw


def _extension_for(raw: bytes, image: Image.Image) -> str:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    fmt = (image.format or "").upper()
    if fmt == "PNG":
        return "png"
    if fmt in ("JPEG", "JPG"):
        return "jpg"
    if fmt == "WEBP":
        return "webp"
    # Always re-encode path uses png
    return "png"


def decode_image_bytes(
    raw: bytes,
    *,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> Image.Image:
    if len(raw) > DEFAULT_MAX_BYTES:
        # caller should already enforce; keep belt-and-suspenders for base64 path
        raise ImageInputError(f"image too large: {len(raw)} bytes")
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        try:
            image = Image.open(BytesIO(raw))
            image.load()
        except Image.DecompressionBombError as err:
            raise ImageInputError(
                f"image too large: exceeds MAX_IMAGE_PIXELS={max_pixels}"
            ) from err
        except UnidentifiedImageError as err:
            raise ImageInputError(
                "invalid image: not a recognized image format (png/jpeg/webp)"
            ) from err
        except OSError as err:
            raise ImageInputError(f"invalid image: {err}") from err

        w, h = image.size
        if w * h > max_pixels:
            raise ImageInputError(
                f"image too large: {w}x{h} ({w * h} px) exceeds max_pixels={max_pixels}"
            )
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    finally:
        Image.MAX_IMAGE_PIXELS = previous


def resolve_image_ref(
    ref: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    timeout_s: float = URL_TIMEOUT_S,
) -> tuple[bytes, str]:
    """Return (raw_bytes, preferred_ext) for a product image ref.

    Accepts https/http URL, data URL, or raw base64.
    """
    if not isinstance(ref, str) or not ref.strip():
        raise ImageInputError("invalid image: empty or missing")

    text = ref.strip()
    # Reject non-http(s) URL schemes before base64 fallback (e.g. file://)
    if "://" in text.split(",", 1)[0] and not text.startswith("data:"):
        scheme = text.split("://", 1)[0].lower()
        if scheme not in ("http", "https"):
            raise ImageInputError(
                f"invalid image URL scheme {scheme!r}; only http/https allowed"
            )

    if text.startswith("http://") or text.startswith("https://"):
        raw = _fetch_url(text, max_bytes=max_bytes, timeout_s=timeout_s)
    elif text.startswith("data:") or _looks_like_base64(text):
        raw = _strip_data_url(text)
        if len(raw) > max_bytes:
            raise ImageInputError(
                f"image too large: {len(raw)} bytes > {max_bytes}"
            )
    else:
        # Treat remaining non-empty as base64 attempt
        raw = _strip_data_url(text)
        if len(raw) > max_bytes:
            raise ImageInputError(
                f"image too large: {len(raw)} bytes > {max_bytes}"
            )

    image = decode_image_bytes(raw, max_pixels=max_pixels)
    ext = _extension_for(raw, image)
    # Normalize to PNG on disk for LoadImage reliability when format was exotic
    if ext not in ("png", "jpg", "webp"):
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue(), "png"
    # If we converted mode, re-encode as PNG so file matches pixels
    if image.mode == "RGB" and ext == "png" and not raw.startswith(b"\x89PNG"):
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue(), "png"
    # For JPEG/WEBP/PNG that opened fine, keep original bytes when possible
    if ext == "png" and raw.startswith(b"\x89PNG"):
        return raw, "png"
    if ext == "jpg" and raw.startswith(b"\xff\xd8\xff"):
        return raw, "jpg"
    if ext == "webp" and len(raw) >= 12 and raw[8:12] == b"WEBP":
        return raw, "webp"
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue(), "png"


def _looks_like_base64(text: str) -> bool:
    compact = "".join(text.split())
    if len(compact) < 16:
        return False
    # Heuristic: long base64 alphabet string
    return bool(re.fullmatch(r"[A-Za-z0-9+/]+=*", compact[:200]))


def stage_image(
    ref: str,
    *,
    input_dir: Path,
    basename_stem: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> str:
    """Resolve ref and write under input_dir. Return basename for LoadImage."""
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    raw, ext = resolve_image_ref(ref, max_bytes=max_bytes, max_pixels=max_pixels)
    # Sanitize stem: only safe chars
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", basename_stem).strip("._") or "frame"
    name = f"{stem}.{ext}"
    path = input_dir / name
    path.write_bytes(raw)
    return name


def cleanup_staged(input_dir: Path, *basenames: str | None) -> None:
    for name in basenames:
        if not name:
            continue
        path = Path(input_dir) / name
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass
