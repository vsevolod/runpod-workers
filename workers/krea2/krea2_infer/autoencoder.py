"""Qwen-Image VAE wrapper (from krea-ai/krea-2, extended for local weights)."""

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
        from safetensors.torch import load_file

        path = Path(vae_path) if vae_path else None
        if path is not None and path.is_file():
            logger.info("Loading VAE from single file %s", path)
            try:
                self.ae = AutoencoderKLQwenImage.from_single_file(str(path))
            except Exception as err:
                logger.warning(
                    "from_single_file failed (%s); falling back to repo config + state_dict",
                    err,
                )
                self.ae = AutoencoderKLQwenImage.from_pretrained(
                    repo_id,
                    subfolder=subfolder,
                    local_files_only=local_files_only,
                )
                state = load_file(str(path))
                missing, unexpected = self.ae.load_state_dict(state, strict=False)
                if missing:
                    logger.warning("VAE missing keys (first 10): %s", missing[:10])
                if unexpected:
                    logger.warning("VAE unexpected keys (first 10): %s", unexpected[:10])
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
