"""Load Krea 2 Turbo (FP8 DiT + Qwen TE + VAE) for resident inference.

Adapted from krea-ai/krea-2 inference.py and modified for FP8 loading and the
RunPod worker pipeline. Licensed under Apache-2.0; see
../LICENSES/KREA-2-APACHE-2.0.txt.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file

from .autoencoder import QwenAutoencoder
from .encoder import Qwen3VLConditioner, TextEncoderConfig
from .lora import LoRACatalog, LoRASelection
from .lora_runtime import LoRAManager, is_fp8_tensor
from .mmdit import SingleMMDiTConfig, SingleStreamDiT
from .sampling import sample

logger = logging.getLogger(__name__)

# Official large-wide MMDiT config from krea-ai/krea-2 inference.py
SINGLE_MMDIT_LARGE_WIDE = SingleMMDiTConfig(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
)

DEFAULT_TEXT_ENCODER_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_VAE_REPO = "Qwen/Qwen-Image"
DEFAULT_VAE_SUBFOLDER = "vae"

DIT_CANDIDATES = (
    "krea2_turbo_fp8.safetensors",
    "krea2_turbo.safetensors",
    "oss_turbo.safetensors",
)
VAE_CANDIDATES = (
    "qwen_image_vae.safetensors",
    "vae.safetensors",
)

def _resolve_path(model_dir: Path, explicit: str | None, candidates: Sequence[str]) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Weight file not found: {path}")
        return path
    for name in candidates:
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    return None


def load_dit(
    dit_path: Path,
    *,
    device: torch.device,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> SingleStreamDiT:
    logger.info("Loading DiT from %s", dit_path)
    with torch.device("meta"):
        mmdit = SingleStreamDiT(SINGLE_MMDIT_LARGE_WIDE)

    state = load_file(str(dit_path), device="cpu")
    has_fp8 = any(is_fp8_tensor(v) for v in state.values() if isinstance(v, torch.Tensor))
    logger.info("DiT state dict tensors: %d (fp8_present=%s)", len(state), has_fp8)

    # AlperKTS / Comfy FP8 exports may include surplus LastLayer tensors
    # (e.g. last.down.weight, last.up.weight) that official SingleStreamDiT
    # does not use. Load non-strict, but fail hard on missing required keys.
    incompatible = mmdit.load_state_dict(state, strict=False, assign=True)
    if incompatible.missing_keys:
        preview = ", ".join(incompatible.missing_keys[:20])
        more = len(incompatible.missing_keys) - 20
        suffix = f" (+{more} more)" if more > 0 else ""
        raise RuntimeError(
            f"Failed to load DiT weights from {dit_path}: "
            f"missing {len(incompatible.missing_keys)} key(s): {preview}{suffix}"
        )
    if incompatible.unexpected_keys:
        logger.warning(
            "Ignoring %d unexpected DiT key(s) (common on community FP8 packs): %s",
            len(incompatible.unexpected_keys),
            incompatible.unexpected_keys,
        )

    if has_fp8:
        # Preserve per-tensor dtypes (FP8 + high-precision norms/biases).
        mmdit = mmdit.to(device=device)
    else:
        mmdit = mmdit.to(device=device, dtype=compute_dtype)

    return mmdit.eval().requires_grad_(False)


def load_autoencoder(
    vae_path: Path | None,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool = False,
) -> QwenAutoencoder:
    logger.info(
        "Loading VAE (local=%s, local_files_only=%s)",
        vae_path,
        local_files_only,
    )
    ae = QwenAutoencoder(
        vae_path=str(vae_path) if vae_path else None,
        local_files_only=local_files_only,
    )
    return ae.to(device=device, dtype=dtype).eval().requires_grad_(False)


def load_text_encoder(
    model_id: str,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool = False,
) -> Qwen3VLConditioner:
    cfg = TextEncoderConfig(model_id=model_id)
    logger.info(
        "Loading text encoder %s (local_files_only=%s)",
        model_id,
        local_files_only,
    )
    encoder = Qwen3VLConditioner(
        cfg.model_id,
        cfg.max_length,
        select_layers=cfg.select_layers,
        local_files_only=local_files_only,
        torch_dtype=dtype,
    )
    return encoder.to(device=device, dtype=dtype).eval().requires_grad_(False)


@dataclass
class Krea2Pipeline:
    dit: SingleStreamDiT
    ae: QwenAutoencoder
    encoder: Qwen3VLConditioner
    device: torch.device
    dtype: torch.dtype = torch.bfloat16
    loras: LoRAManager | None = None

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        width: int = 1024,
        height: int = 1024,
        steps: int = 8,
        guidance: float = 0.0,
        seed: int = 0,
        mu: float | None = 1.15,
        num_images: int = 1,
        negative_prompt: str | None = None,
        loras: Sequence[LoRASelection] = (),
    ):
        prompts = [prompt] * num_images
        negatives = None
        if negative_prompt is not None and guidance > 0:
            negatives = [negative_prompt] * num_images

        if loras and self.loras is None:
            raise RuntimeError("LoRA manager is not configured")
        lora_activation = self.loras.activation(loras) if loras else None
        return sample(
            self.dit,
            self.ae,
            self.encoder,
            prompts,
            negative_prompts=negatives,
            device=str(self.device),
            dtype=self.dtype,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            seed=seed,
            mu=mu,
            lora_activation=lora_activation,
        )


def load_pipeline(
    model_dir: str | Path | None = None,
    *,
    dit_path: str | None = None,
    vae_path: str | None = None,
    text_encoder_id: str | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool | None = None,
    lora_dir: str | Path | None = None,
) -> Krea2Pipeline:
    """Load TE + DiT + VAE once (TE is offloaded to CPU after each encode)."""
    model_dir = Path(model_dir or os.environ.get("MODEL_DIR", "/runpod-volume/krea2"))
    lora_dir = Path(lora_dir or os.environ.get("LORA_DIR", "") or model_dir / "loras")
    text_encoder_id = text_encoder_id or os.environ.get(
        "TEXT_ENCODER_ID", DEFAULT_TEXT_ENCODER_ID
    )
    if local_files_only is None:
        local_files_only = os.environ.get("LOCAL_FILES_ONLY", "").lower() in {
            "1",
            "true",
            "yes",
        }

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    resolved_dit = _resolve_path(
        model_dir, dit_path or os.environ.get("DIT_PATH"), DIT_CANDIDATES
    )
    if resolved_dit is None:
        raise FileNotFoundError(
            f"DiT weights not found under {model_dir}. Expected one of: "
            f"{', '.join(DIT_CANDIDATES)}. Set DIT_PATH or MODEL_DIR."
        )

    resolved_vae = _resolve_path(
        model_dir, vae_path or os.environ.get("VAE_PATH"), VAE_CANDIDATES
    )
    if resolved_vae is None:
        logger.warning(
            "No local VAE safetensors in %s; will load %s/%s from Hugging Face",
            model_dir,
            DEFAULT_VAE_REPO,
            DEFAULT_VAE_SUBFOLDER,
        )

    dit = load_dit(resolved_dit, device=device, compute_dtype=dtype)
    lora_catalog = LoRACatalog.scan(lora_dir)
    lora_manager = LoRAManager(dit, lora_catalog, dtype)
    ae = load_autoencoder(
        resolved_vae, device=device, dtype=dtype, local_files_only=local_files_only
    )
    encoder = load_text_encoder(
        text_encoder_id,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
    )

    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        logger.info(
            "Models loaded. CUDA mem free/total: %.2f / %.2f GiB",
            free / (1024**3),
            total / (1024**3),
        )

    return Krea2Pipeline(
        dit=dit,
        ae=ae,
        encoder=encoder,
        device=device,
        loras=lora_manager,
        dtype=dtype,
    )
