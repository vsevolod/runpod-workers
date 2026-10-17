"""Base64 image input and PNG delivery shared by image workers.

Portions adapted from runpod-workers/worker-sdxl; see
LICENSES/RUNPOD-WORKER-SDXL-MIT.txt.
"""

import base64
from io import BytesIO
import os
from pathlib import Path
import tempfile

from PIL import Image


def decode_image_string(value, *, preserve_alpha=False, max_bytes=None, max_side=None):
    # Import locally to keep the shared request-error type without an import cycle.
    from .contract import RequestError

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
    if max_bytes is not None and len(raw) > max_bytes:
        raise RequestError(f"images[] entry is {len(raw)} bytes; limit is {max_bytes}")
    try:
        image = Image.open(BytesIO(raw))
        image.load()
    except Exception as err:
        raise RequestError("images[] could not be decoded as an image") from err
    if max_side is not None and max(image.size) > max_side:
        raise RequestError(f"images[] side must be <= {max_side}px (got {image.width}x{image.height})")
    alpha = image.mode in ("RGBA", "LA", "PA") or image.mode == "P" and "transparency" in image.info
    return image.convert("RGBA" if preserve_alpha and alpha else "RGB")


def images_to_base64_data_urls(images):
    urls = []
    for image in images:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        urls.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
    return urls


def save_and_upload_images(images, job_id):
    if not os.environ.get("BUCKET_ENDPOINT_URL"):
        return images_to_base64_data_urls(images)
    from runpod.serverless.utils import rp_upload

    urls = []
    with tempfile.TemporaryDirectory(prefix="runpod-images-") as directory:
        for index, image in enumerate(images):
            path = Path(directory) / f"{index}.png"
            image.save(path, format="PNG")
            urls.append(rp_upload.upload_image(job_id, str(path)))
    return urls
