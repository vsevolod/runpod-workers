"""Load Krea 2 Turbo (FP8 or INT8-ConvRot DiT + Qwen TE + VAE) for resident inference.

Adapted from krea-ai/krea-2 inference.py and modified for FP8 / INT8 loading and the
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

from PIL import Image

from .autoencoder import QwenAutoencoder
from .dit_quant import (
    MODE_FP8,
    MODE_INT8_CONVROT,
    dit_candidates_for,
    resolve_dit_quant,
)
from .edit_sampling import sample_edit
from .encoder import Qwen3VLConditioner, TextEncoderConfig
from .int8_linear import (
    apply_int8_side_tensors,
    partition_int8_state_dict,
    state_dict_has_int8_tensorwise,
)
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

# Back-compat alias (FP8 / default path filenames).
DIT_CANDIDATES = tuple(dit_candidates_for(MODE_FP8))
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


def _load_state_dict_into_dit(
    mmdit: SingleStreamDiT,
    state: dict[str, torch.Tensor],
    dit_path: Path,
) -> None:
    # AlperKTS / Comfy exports may include surplus LastLayer tensors
    # (e.g. last.down.weight, last.up.weight) that official SingleStreamDiT
    # does not use. Load non-strict, but fail hard on missing required keys.
    #
    # INT8: assign=True replaces Parameter data with int8 tensors. nn.Linear
    # defaults to requires_grad=True, and PyTorch rejects non-floating
    # Parameters that require grad ("Only Tensors of floating point and
    # complex dtype can require gradients"). Inference never needs grad.
    mmdit.requires_grad_(False)
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
            "Ignoring %d unexpected DiT key(s) (common on community packs): %s",
            len(incompatible.unexpected_keys),
            incompatible.unexpected_keys,
        )


def load_dit(
    dit_path: Path,
    *,
    device: torch.device,
    compute_dtype: torch.dtype = torch.bfloat16,
    quant_mode: str | None = None,
) -> SingleStreamDiT:
    """Load DiT weights.

    ``quant_mode`` (or env ``DIT_QUANT``):
      * ``fp8`` (default) — FP8/BF16 storage; FP8 Linear cast to compute dtype.
      * ``int8_convrot`` — int8_tensorwise + online ConvRot; INT8 GEMM when available.
    """
    mode = resolve_dit_quant(quant_mode)
    logger.info("Loading DiT from %s (DIT_QUANT=%s)", dit_path, mode)
    with torch.device("meta"):
        mmdit = SingleStreamDiT(SINGLE_MMDIT_LARGE_WIDE)

    state = load_file(str(dit_path), device="cpu")
    has_fp8 = any(is_fp8_tensor(v) for v in state.values() if isinstance(v, torch.Tensor))
    has_int8 = state_dict_has_int8_tensorwise(state)
    logger.info(
        "DiT state dict tensors: %d (fp8_present=%s, int8_tensorwise=%s)",
        len(state),
        has_fp8,
        has_int8,
    )

    if mode == MODE_INT8_CONVROT:
        if not has_int8:
            raise RuntimeError(
                f"DIT_QUANT={MODE_INT8_CONVROT} but {dit_path} has no int8_tensorwise "
                f"weights (expected int8 .weight + .weight_scale). "
                f"Place krea2_turbo_int8_convrot.safetensors or set DIT_PATH."
            )
        weight_state, side = partition_int8_state_dict(state)
        _load_state_dict_into_dit(mmdit, weight_state, dit_path)
        n_int8 = apply_int8_side_tensors(mmdit, side, weight_state)
        logger.info("Configured %d INT8 tensorwise Linear layer(s)", n_int8)
        if n_int8 == 0:
            raise RuntimeError(
                f"DIT_QUANT={MODE_INT8_CONVROT} but no Linear layers received int8 weights"
            )
        # Preserve per-tensor dtypes (int8 weights + fp norms/biases).
        mmdit = mmdit.to(device=device)
    else:
        if has_int8:
            raise RuntimeError(
                f"DIT_QUANT={MODE_FP8} but {dit_path} looks like int8_tensorwise. "
                f"Set DIT_QUANT=int8_convrot to enable the INT8 ConvRot path."
            )
        _load_state_dict_into_dit(mmdit, state, dit_path)
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
    quant_mode: str = MODE_FP8

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

    @torch.inference_mode()
    def edit(
        self,
        prompt: str,
        sources: Sequence[Image.Image] | None = None,
        *,
        source: Image.Image | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int = 8,
        guidance: float = 0.0,
        seed: int = 0,
        mu: float | None = 1.15,
        negative_prompt: str | None = None,
        grounding_px: int = 768,
        ref_boost: float = 1.0,
        fit_mode: str = "fit",
        loras: Sequence[LoRASelection] = (),
    ):
        """Identity edit with 1..2 reference images.

        Prefer ``sources=``. Deprecated ``source=`` alone is accepted as a
        single-ref adapter. Passing both is an error.
        """
        if sources is not None and source is not None:
            raise ValueError("pass only one of source= or sources=")
        if sources is None and source is None:
            raise ValueError("edit requires sources= or source=")
        if sources is None:
            resolved: Sequence[Image.Image] = (source,)
        else:
            resolved = sources
        if loras and self.loras is None:
            raise RuntimeError("LoRA manager is not configured")
        # Edit never uses resolution-auto mu (would depend on wrong seq_len).
        if mu is None:
            mu = 1.15
        lora_activation = self.loras.activation(loras) if loras else None
        return sample_edit(
            self.dit,
            self.ae,
            self.encoder,
            prompt,
            resolved,
            negative_prompt=negative_prompt,
            device=str(self.device),
            dtype=self.dtype,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            seed=seed,
            mu=mu,
            grounding_px=grounding_px,
            ref_boost=ref_boost,
            fit_mode=fit_mode,
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
    quant_mode: str | None = None,
) -> Krea2Pipeline:
    """Load TE + DiT + VAE once (TE is offloaded to CPU after each encode).

    DiT quant path is selected by ``quant_mode`` or env ``DIT_QUANT``
    (``fp8`` default, ``int8_convrot`` for INT8 ConvRot).
    """
    model_dir = Path(model_dir or os.environ.get("MODEL_DIR", "/runpod-volume/krea2"))
    lora_dir = Path(lora_dir or os.environ.get("LORA_DIR", "") or model_dir / "loras")
    text_encoder_id = text_encoder_id or os.environ.get(
        "TEXT_ENCODER_ID", DEFAULT_TEXT_ENCODER_ID
    )
    mode = resolve_dit_quant(quant_mode)
    dit_candidates = dit_candidates_for(mode)

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
        model_dir, dit_path or os.environ.get("DIT_PATH"), dit_candidates
    )
    if resolved_dit is None:
        raise FileNotFoundError(
            f"DiT weights not found under {model_dir} for DIT_QUANT={mode}. "
            f"Expected one of: {', '.join(dit_candidates)}. Set DIT_PATH or MODEL_DIR."
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

    logger.info("Resolved DiT path=%s quant_mode=%s", resolved_dit, mode)
    dit = load_dit(
        resolved_dit, device=device, compute_dtype=dtype, quant_mode=mode
    )
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
        quant_mode=mode,
    )
