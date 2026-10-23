"""Shared model/VRAM megapixel budgets with explicitly documented calibration."""

from dataclasses import dataclass
import logging
import math

from PIL import Image

logger = logging.getLogger(__name__)

# Minimum nominal VRAM class in GB -> decimal megapixels.
# Krea/Qwen: 48 GB calibrated on L40S with Identity Edit V2 / Lenovo.
# MOSS Base/Turbo: 24 GB (RTX 4090) and 48 GB (L40S), one square output,
# CPU offload, through the API ceiling of 2048**2 pixels = 4.194304 MP.
# All other entries remain 1 MP placeholders, not measured hardware limits.
# Protocol, raw measurements and scope: docs/benchmarks/image-megapixels.md.
MEGAPIXEL_LIMITS = {
    # Conservative total reference budget for MOSS Edit; not GPU calibrated.
    "moss_image.edit": {0: .262144},
    # Fibo native 1024-square default; no GPU capacity calibration yet.
    "fibo_2.base": {0: 1.048576},
    "fibo_2.turbo": {0: 1.048576},
    "moss_image.base": {0: 1., 8: 1., 12: 1., 16: 1., 24: 4.194304, 32: 1., 48: 4.194304, 64: 1., 80: 1., 96: 1., 192: 1.},
    "moss_image.turbo": {0: 1., 8: 1., 12: 1., 16: 1., 24: 4.194304, 32: 1., 48: 4.194304, 64: 1., 80: 1., 96: 1., 192: 1.},
    "qwen_image_21.base": {0: 1., 8: 1., 12: 1., 16: 1., 24: 1., 32: 1., 48: 4., 64: 1., 80: 1., 96: 1., 192: 1.},
    "qwen_image_21.turbo": {0: 1., 8: 1., 12: 1., 16: 1., 24: 1., 32: 1., 48: 4., 64: 1., 80: 1., 96: 1., 192: 1.},
    "krea2.fp8": {0: 1., 8: 1., 12: 1., 16: 1., 24: 1., 32: 1., 48: 4., 64: 1., 80: 1., 96: 1., 192: 1.},
    "krea2.int8_convrot": {0: 1., 8: 1., 12: 1., 16: 1., 24: 1., 32: 1., 48: 4., 64: 1., 80: 1., 96: 1., 192: 1.},
}


@dataclass(frozen=True)
class MegapixelBudget:
    megapixels: float
    vram_gib: float | None


def detect_vram_gib() -> float | None:
    """Total memory of the active CUDA device; never free/remaining VRAM."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3


def resolve_megapixels(requested: float | str, model: str) -> MegapixelBudget:
    if requested != "max":
        return MegapixelBudget(float(requested), None)
    vram = detect_vram_gib()
    # Table classes are nominal GB; CUDA diagnostics stay in binary GiB.
    # L40S reports 47,677,177,856 bytes = 44.40 GiB = 47.68 GB -> class 48.
    # Round up the final partial GB to account for device reservations.
    nominal = math.ceil(vram * 1024**3 / 1_000_000_000) if vram is not None else 0
    row = MEGAPIXEL_LIMITS.get(model, {0: 1.})
    bucket = max((size for size in row if size <= nominal), default=0)
    megapixels = row.get(bucket, 1.)
    logger.info("megapixel_budget requested=max resolved=%g model=%s vram_gib=%s bucket=%s",
                megapixels, model, vram, bucket)
    return MegapixelBudget(megapixels, vram)


def canvas_at_megapixels(image: Image.Image, megapixels: float, *, multiple: int) -> tuple[int, int]:
    """Preserve the original aspect, including upscaling, with floor alignment."""
    area = int(megapixels * 1_000_000)
    ratio = image.width / image.height
    return (max(multiple, int(math.sqrt(area * ratio)) // multiple * multiple),
            max(multiple, int(math.sqrt(area / ratio)) // multiple * multiple))


def resize_references(images, megapixels: float, *, multiple: int, max_side: int):
    if not images:
        return ()
    pixels = int(megapixels * 1_000_000) // len(images)
    if pixels < multiple**2:
        raise ValueError(f"reference_megapixels must be at least {len(images) * multiple**2 / 1_000_000:g} "
                         f"for {len(images)} references (each requires at least {multiple}x{multiple} pixels)")
    result = []
    for image in images:
        width, height = image.size
        scale = min(1., math.sqrt(pixels / (width * height)), max_side / width, max_side / height,
                    pixels / (multiple * width), pixels / (multiple * height))
        size = tuple(max(multiple, int(side * scale) // multiple * multiple) for side in image.size)
        result.append(image if size == image.size else image.resize(size, Image.Resampling.LANCZOS))
    return tuple(result)
