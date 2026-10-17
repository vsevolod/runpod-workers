"""Load Qwen-Image-2.1 once and run text-to-image or multi-reference edit."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

workers_dir = Path(__file__).resolve().parents[2]
if (workers_dir / "shared").is_dir():
    sys.path.insert(0, str(workers_dir))
from shared.prefetch import wait_for_component

from .diagnostics import log_cuda_info, log_runtime_info, log_stage
from .lora import LoRACatalog, LoRAManager
from .request import NormalizedRequest, build_pipe_kwargs
from .runtime_config import (
    DEFAULT_MODEL_DIR,
    LORA_DIR_NAME,
    TEXTURE_FIX_VAE_DIR,
    TEXTURE_FIX_VAE_ID,
    parse_disable_mmap,
    parse_local_files_only,
    resolve_model_source,
    resolve_offload,
    resolve_vae_variant,
    validate_turbo_config,
)

logger = logging.getLogger(__name__)


class QwenImage21Runtime:
    """Resident Diffusers pipeline. One instance per worker process."""

    def __init__(self, pipe, lora_catalog: LoRACatalog, *, offload: str | None = None):
        self.pipe = pipe
        self.loras = LoRAManager(pipe, lora_catalog, offload=offload)

    def run(self, norm: NormalizedRequest, seed: int, loras=()):
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(int(seed))
        kwargs = build_pipe_kwargs(norm)
        kwargs["generator"] = generator
        with self.loras.activate(loras):
            result = self.pipe(**kwargs)
        images = list(result.images)
        if not images:
            raise RuntimeError("Qwen-Image-2.1 pipeline returned no images")
        return images


def load_pipeline(
    model_dir: str | None = None,
    model_id: str | None = None,
    offload: str | None = None,
) -> QwenImage21Runtime:
    """Load ``Qwen/Qwen-Image-2.1-Turbo`` in bf16 and place it according to ``OFFLOAD``."""
    with log_stage("runtime_imports"):
        import torch
        from diffusers import QwenImage21Pipeline

    log_runtime_info(torch)
    with log_stage("cuda_probe"):
        log_cuda_info(torch)

    # Imports/probing run while the separate process fetches the snapshot.
    # Preserve mmap and CPU offload; do not inspect a partial model directory.
    wait_for_component("all")
    with log_stage("model_source"):
        source = resolve_model_source(model_dir, model_id)
        local_files_only = parse_local_files_only()
        mode = resolve_offload(offload)
        disable_mmap = parse_disable_mmap()
        vae_variant = resolve_vae_variant()
        directory = Path(
            model_dir if model_dir is not None else os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)
        )
        vae_source = None
        if vae_variant == "texture-fix":
            vae_directory = directory / TEXTURE_FIX_VAE_DIR
            vae_source = (
                str(vae_directory)
                if local_files_only or (vae_directory / "config.json").is_file()
                else TEXTURE_FIX_VAE_ID
            )
        lora_directory = Path(os.environ.get("LORA_DIR") or directory / LORA_DIR_NAME)
    with log_stage("lora_catalog"):
        lora_catalog = LoRACatalog.scan(lora_directory)
    logger.info(
        "Loading Qwen-Image-2.1-Turbo from %s (offload=%s, local_files_only=%s, disable_mmap=%s)",
        source,
        mode,
        local_files_only,
        disable_mmap,
    )
    with log_stage("base_model_load"):
        pipe = _load_base_pipeline(
            QwenImage21Pipeline,
            source,
            local_files_only=local_files_only,
            disable_mmap=disable_mmap,
            vae_source=vae_source,
        )
    # Keep both the scheduler and sample_sigmas supplied by the checkpoint.
    # Older Diffusers drops sample_sigmas at load time, which must fail visibly.
    with log_stage("turbo_config"):
        validate_turbo_config(pipe.config, source)
    pipe.set_progress_bar_config(disable=True)

    with log_stage(f"placement_{mode}"):
        if mode == "cpu":
            pipe.enable_model_cpu_offload()
        elif mode == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("OFFLOAD=none requires a CUDA device")
            pipe.to("cuda")

    logger.info(
        "Qwen-Image-2.1-Turbo ready (offload=%s, steps=%s, vae=%s)",
        mode, len(pipe.config["sample_sigmas"]), vae_variant,
    )
    return QwenImage21Runtime(pipe, lora_catalog, offload=mode)


def _load_base_pipeline(
    pipe_class, source: str, *, local_files_only: bool, disable_mmap: bool,
    vae_source: str | None = None,
):
    import torch

    components = {}
    if vae_source is not None:
        from diffusers import AutoencoderKLQwenImage21

        with log_stage("vae_load"):
            logger.info("Loading texture-fix VAE from %s", vae_source)
            components["vae"] = AutoencoderKLQwenImage21.from_pretrained(
                vae_source,
                torch_dtype=torch.bfloat16,
                local_files_only=local_files_only,
                disable_mmap=disable_mmap,
            )
    if disable_mmap:
        from transformers import Qwen3VLForConditionalGeneration

        # Diffusers forwards disable_mmap only to ModelMixin components, not
        # Transformers models. Preload the encoder so all base weights opt out.
        with log_stage("text_encoder_load"):
            components["text_encoder"] = Qwen3VLForConditionalGeneration.from_pretrained(
                source,
                subfolder="text_encoder",
                dtype=torch.bfloat16,
                local_files_only=local_files_only,
                disable_mmap=True,
            )
    with log_stage("pipeline_components_load"):
        return pipe_class.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
            disable_mmap=disable_mmap,
            **components,
        )
