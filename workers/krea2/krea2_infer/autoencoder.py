"""Qwen-Image VAE wrapper.

Adapted from krea-ai/krea-2 and modified to support local VAE weights.
Licensed under Apache-2.0; see ../LICENSES/KREA-2-APACHE-2.0.txt.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from einops import rearrange
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class QwenAutoencoder(nn.Module):
    """qwen-ae-f8-16c: the Qwen-Image VAE (f8, 16 latent channels)."""

    def __init__(
        self,
        vae_path: str | None = None,
        *,
        local_files_only: bool = False,
        repo_id: str = "Qwen/Qwen-Image",
        subfolder: str = "vae",
    ):
        super().__init__()
        from diffusers import AutoencoderKLQwenImage

        path = Path(vae_path) if vae_path else None
        if path is not None and path.is_file():
            logger.info("Loading VAE from single file %s", path)
            try:
                self.ae = AutoencoderKLQwenImage.from_single_file(str(path))
            except Exception as err:
                # Local Comfy/community packs often use non-Diffusers key names
                # (e.g. encoder.downsamples.*). Do not overlay them onto a HF
                # config with strict=False — that silently corrupts the VAE.
                logger.warning(
                    "from_single_file failed for %s (%s); ignoring local file "
                    "and loading clean VAE from %s/%s",
                    path,
                    err,
                    repo_id,
                    subfolder,
                )
                self.ae = AutoencoderKLQwenImage.from_pretrained(
                    repo_id,
                    subfolder=subfolder,
                    local_files_only=local_files_only,
                )
        else:
            logger.info(
                "Loading VAE from Hugging Face %s/%s (local_files_only=%s)",
                repo_id,
                subfolder,
                local_files_only,
            )
            self.ae = AutoencoderKLQwenImage.from_pretrained(
                repo_id,
                subfolder=subfolder,
                local_files_only=local_files_only,
            )

        self.compression = 8
        self.channels = 16
        self.register_buffer(
            "latents_mean",
            torch.tensor(self.ae.latents_mean).view(1, -1, 1, 1, 1),
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(self.ae.latents_std).view(1, -1, 1, 1, 1),
        )

    def decode(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b c h w -> b c 1 h w")
        x = (x * self.latents_std) + self.latents_mean
        return rearrange(self.ae.decode(x).sample, "b c 1 h w -> b c h w")
