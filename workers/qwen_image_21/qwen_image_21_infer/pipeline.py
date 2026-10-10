"""Load Qwen-Image-2.1 once and run text-to-image or multi-reference edit."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .diagnostics import log_cuda_info, log_runtime_info, log_stage
from .lora import LoRACatalog, LoRAManager, TURBO_ADAPTER_NAME
from .request import NormalizedRequest, build_pipe_kwargs
from .runtime_config import (
    DEFAULT_MODEL_DIR,
    LORA_DIR_NAME,
    parse_local_files_only,
    resolve_lora_source,
    resolve_model_source,
    resolve_offload,
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
    """Load ``Qwen/Qwen-Image-2.1`` in bf16 and place it according to ``OFFLOAD``."""
    with log_stage("runtime_imports"):
        import torch
        from diffusers import QwenImage21Pipeline

    log_runtime_info(torch)
    with log_stage("cuda_probe"):
        log_cuda_info(torch)

    with log_stage("model_source"):
        source = resolve_model_source(model_dir, model_id)
        local_files_only = parse_local_files_only()
        mode = resolve_offload(offload)
        directory = Path(
            model_dir if model_dir is not None else os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)
        )
        lora_directory = Path(os.environ.get("LORA_DIR") or directory / LORA_DIR_NAME)
    with log_stage("lora_catalog"):
        lora_catalog = LoRACatalog.scan(lora_directory)
    logger.info(
        "Loading Qwen-Image-2.1 from %s (offload=%s, local_files_only=%s)",
        source,
        mode,
        local_files_only,
    )
    with log_stage("base_model_load"):
        pipe = QwenImage21Pipeline.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        )
    _load_turbo_lora(pipe, model_dir, local_files_only=local_files_only)
    with log_stage("turbo_scheduler"):
        _use_turbo_scheduler(pipe)
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

    logger.info("Qwen-Image-2.1 ready (offload=%s, viggle_turbo=v0.2.1)", mode)
    return QwenImage21Runtime(pipe, lora_catalog, offload=mode)


def _load_turbo_lora(pipe, model_dir: str | None, *, local_files_only: bool) -> None:
    """Attach the Viggle adapter at runtime. Do not merge it into bf16 weights."""
    with log_stage("turbo_lora_source"):
        source, weight_name, files_only = resolve_lora_source(
            model_dir, local_files_only=local_files_only
        )
    logger.info("Loading Viggle turbo LoRA %s from %s", weight_name, source)
    with log_stage("turbo_lora_load"):
        pipe.load_lora_weights(
            source,
            weight_name=weight_name,
            local_files_only=files_only,
            adapter_name=TURBO_ADAPTER_NAME,
        )
    with log_stage("turbo_lora_activate"):
        pipe.set_adapters([TURBO_ADAPTER_NAME], adapter_weights=[1.0])


def _use_turbo_scheduler(pipe) -> None:
    """The base config's shift_terminal=0.02 wrecks the turbo schedule's last step."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        shift_terminal=None,
    )
