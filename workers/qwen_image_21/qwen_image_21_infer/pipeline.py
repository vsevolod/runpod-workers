"""Load Qwen-Image-2.1 once and run text-to-image or multi-reference edit."""

from __future__ import annotations

import logging

from .request import NormalizedRequest, build_pipe_kwargs
from .runtime_config import (
    parse_local_files_only,
    resolve_lora_source,
    resolve_model_source,
    resolve_offload,
)

logger = logging.getLogger(__name__)


class QwenImage21Runtime:
    """Resident Diffusers pipeline. One instance per worker process."""

    def __init__(self, pipe):
        self.pipe = pipe

    def run(self, norm: NormalizedRequest, seed: int):
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(int(seed))
        kwargs = build_pipe_kwargs(norm)
        kwargs["generator"] = generator
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
    import torch
    from diffusers import QwenImage21Pipeline

    source = resolve_model_source(model_dir, model_id)
    local_files_only = parse_local_files_only()
    mode = resolve_offload(offload)
    logger.info(
        "Loading Qwen-Image-2.1 from %s (offload=%s, local_files_only=%s)",
        source,
        mode,
        local_files_only,
    )
    pipe = QwenImage21Pipeline.from_pretrained(
        source,
        torch_dtype=torch.bfloat16,
        local_files_only=local_files_only,
    )
    _load_turbo_lora(pipe, model_dir, local_files_only=local_files_only)
    _use_turbo_scheduler(pipe)
    pipe.set_progress_bar_config(disable=True)

    if mode == "cpu":
        pipe.enable_model_cpu_offload()
    elif mode == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("OFFLOAD=none requires a CUDA device")
        pipe.to("cuda")

    logger.info("Qwen-Image-2.1 ready (offload=%s, viggle_turbo=v0.2.1)", mode)
    return QwenImage21Runtime(pipe)


def _load_turbo_lora(pipe, model_dir: str | None, *, local_files_only: bool) -> None:
    """Attach the Viggle adapter at runtime. Do not merge it into bf16 weights."""
    source, weight_name, files_only = resolve_lora_source(
        model_dir, local_files_only=local_files_only
    )
    logger.info("Loading Viggle turbo LoRA %s from %s", weight_name, source)
    pipe.load_lora_weights(
        source,
        weight_name=weight_name,
        local_files_only=files_only,
    )


def _use_turbo_scheduler(pipe) -> None:
    """The base config's shift_terminal=0.02 wrecks the turbo schedule's last step."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        shift_terminal=None,
    )
