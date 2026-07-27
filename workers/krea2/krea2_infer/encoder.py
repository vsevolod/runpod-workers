"""Qwen3-VL text conditioner.

Adapted from krea-ai/krea-2 and modified for local cache/model loading.
Licensed under Apache-2.0; see ../LICENSES/KREA-2-APACHE-2.0.txt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from PIL import Image
from torch import Tensor
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2TokenizerFast,
    Qwen3VLForConditionalGeneration,
)

# Re-export grounded helpers (identity-edit path; implementation in grounded.py).
from .grounded import (  # noqa: E402
    GROUNDED_TEMPLATE,
    grounded_encode_impl,
    grounded_template,
    prepare_grounded_images,
    resize_for_grounding,
)

__all__ = [
    "GROUNDED_TEMPLATE",
    "TextEncoderConfig",
    "Qwen3VLConditioner",
    "grounded_encode_impl",
    "grounded_template",
    "prepare_grounded_images",
    "resize_for_grounding",
]


@dataclass
class TextEncoderConfig:
    model_id: str
    max_length: int = 512
    select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


class Qwen3VLConditioner(torch.nn.Module):
    def __init__(
        self,
        version: str,
        max_length: int = 512,
        select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35),
        *,
        local_files_only: bool = False,
        torch_dtype: torch.dtype | None = torch.bfloat16,
    ):
        super().__init__()
        load_kwargs = {
            "local_files_only": local_files_only,
            "torch_dtype": torch_dtype,
        }
        self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            version, **load_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            version, max_length=max_length, local_files_only=local_files_only
        )
        # Text-only path uses Qwen2TokenizerFast for suffix tokenization.
        # Must NOT reassign self.processor to AutoProcessor (breaks generate).
        self.processor = Qwen2TokenizerFast.from_pretrained(
            version, max_length=max_length, local_files_only=local_files_only
        )
        # Multimodal grounded encode (image_edit) uses AutoProcessor separately.
        self.mm_processor = AutoProcessor.from_pretrained(
            version, local_files_only=local_files_only
        )
        self.qwen = self.qwen.eval().requires_grad_(False)
        self.max_length = max_length
        self.select_layers = select_layers
        self.prompt_template_encode_prefix = (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
            "texture, quantity, text, spatial relationships of the objects and background:"
            "<|im_end|>\n<|im_start|>user\n"
        )
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.prompt_template_encode_suffix_start_idx = 5

    def forward(self, text: list[str]) -> tuple[Tensor, Tensor]:
        prefix_idx = self.prompt_template_encode_start_idx
        text = [self.prompt_template_encode_prefix + item for item in text]
        suffix_text = [self.prompt_template_encode_suffix] * len(text)
        suffix_inputs = self.processor(text=suffix_text, return_tensors="pt").to(
            self.qwen.device, non_blocking=True
        )
        suffix_ids, suffix_mask = (
            suffix_inputs["input_ids"],
            suffix_inputs["attention_mask"].bool(),
        )

        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                max_length=self.max_length
                + prefix_idx
                - self.prompt_template_encode_suffix_start_idx,
                return_tensors="pt",
            ).to(self.qwen.device, non_blocking=True)
            input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
            mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)
            states = self.qwen(
                input_ids=input_ids, attention_mask=mask, output_hidden_states=True
            )

            hiddens = torch.stack(
                [states.hidden_states[i] for i in self.select_layers], dim=2
            )
            hiddens = hiddens[:, prefix_idx:]
            mask = mask[:, prefix_idx:]

            return hiddens, mask

    def grounded_encode(
        self,
        text: str,
        images: Sequence[Image.Image],
        *,
        grounding_px: int = 768,
    ) -> tuple[Tensor, Tensor]:
        """
        Multimodal encode for identity edit: one instruction + 1..2 images.

        Returns:
          hiddens: (1, seq_after_prefix, len(select_layers), hidden_dim)
          mask:    (1, seq_after_prefix) bool
        Batch is always 1 (caller loops for CFG).
        """
        return grounded_encode_impl(
            text,
            images,
            grounding_px=grounding_px,
            mm_processor=self.mm_processor,
            qwen=self.qwen,
            select_layers=self.select_layers,
            prefix_idx=self.prompt_template_encode_start_idx,
        )
