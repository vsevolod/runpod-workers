"""Grounded multimodal TE helpers for identity edit (1..2 images).

Pure-ish utilities used by the Qwen3-VL conditioner. Kept free of
``transformers`` so unit tests can load this module with torch + PIL only.
"""

from __future__ import annotations

from typing import Sequence

import torch
from PIL import Image
from torch import Tensor

_GROUNDED_SYSTEM = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n"
)
_VISION_SLOT = "<|vision_start|><|image_pad|><|vision_end|>"

# Back-compat single-image format string.
GROUNDED_TEMPLATE = (
    _GROUNDED_SYSTEM
    + "<|im_start|>user\n"
    + _VISION_SLOT
    + "{instruction}<|im_end|>\n<|im_start|>assistant\n"
)


def grounded_template(instruction: str, num_images: int = 1) -> str:
    """Build grounded chat template with 1 or 2 vision slots before instruction."""
    if num_images not in (1, 2):
        raise ValueError(f"grounded_template supports 1 or 2 images (got {num_images})")
    vision = _VISION_SLOT * num_images
    return (
        f"{_GROUNDED_SYSTEM}"
        f"<|im_start|>user\n{vision}"
        f"{instruction or ''}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def resize_for_grounding(image: Image.Image, grounding_px: int) -> Image.Image:
    img = image.convert("RGB")
    if not grounding_px:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= grounding_px:
        return img
    s = grounding_px / m
    nw, nh = max(16, round(w * s)), max(16, round(h * s))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def prepare_grounded_images(
    images: Sequence[Image.Image],
    grounding_px: int,
) -> list[Image.Image]:
    """Resize 1..2 RGB images for multimodal TE (order preserved)."""
    if not isinstance(images, (list, tuple)):
        raise ValueError("images must be a sequence of PIL images")
    n = len(images)
    if n not in (1, 2):
        raise ValueError(f"grounded encode requires 1 or 2 images (got {n})")
    return [resize_for_grounding(im, grounding_px) for im in images]


def grounded_encode_impl(
    text: str,
    images: Sequence[Image.Image],
    *,
    grounding_px: int,
    mm_processor,
    qwen,
    select_layers: Sequence[int],
    prefix_idx: int,
) -> tuple[Tensor, Tensor]:
    """
    Multimodal encode: one instruction + 1..2 images.

    Returns:
      hiddens: (1, seq_after_prefix, len(select_layers), hidden_dim)
      mask:    (1, seq_after_prefix) bool
    """
    if not isinstance(text, str):
        raise ValueError("grounded_encode text must be a string")
    prepared = prepare_grounded_images(images, grounding_px)
    if mm_processor is None:
        raise RuntimeError("mm_processor (AutoProcessor) is required for image_edit")

    prompt = grounded_template(text, num_images=len(prepared))
    inputs = mm_processor(
        text=[prompt],
        images=prepared,
        padding=True,
        return_tensors="pt",
    )
    device = next(qwen.parameters()).device
    te_kwargs = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs.get("attention_mask"),
        "pixel_values": inputs.get("pixel_values"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "output_hidden_states": True,
    }
    if te_kwargs["attention_mask"] is not None:
        te_kwargs["attention_mask"] = te_kwargs["attention_mask"].to(device)
    if te_kwargs["pixel_values"] is not None:
        te_kwargs["pixel_values"] = te_kwargs["pixel_values"].to(device)
    if te_kwargs["image_grid_thw"] is not None:
        te_kwargs["image_grid_thw"] = te_kwargs["image_grid_thw"].to(device)
    if inputs.get("mm_token_type_ids") is not None:
        te_kwargs["mm_token_type_ids"] = inputs["mm_token_type_ids"].to(device)

    with torch.no_grad():
        states = qwen(**te_kwargs)
        hiddens = torch.stack(
            [states.hidden_states[i] for i in select_layers], dim=2
        )
    attn = te_kwargs.get("attention_mask")
    if attn is None:
        mask = torch.ones(hiddens.shape[:2], device=hiddens.device, dtype=torch.bool)
    else:
        mask = attn.bool()
    return hiddens[:, prefix_idx:], mask[:, prefix_idx:]
