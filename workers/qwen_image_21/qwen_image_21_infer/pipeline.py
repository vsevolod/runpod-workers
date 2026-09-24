"""Load Qwen-Image-2.1 once and run text-to-image or multi-reference edit."""

from __future__ import annotations

import logging

from .request import NormalizedRequest, build_pipe_kwargs
from .runtime_config import parse_local_files_only, resolve_model_source, resolve_offload

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
    pipe.set_progress_bar_config(disable=True)

    if mode == "cpu":
        pipe.enable_model_cpu_offload()
    elif mode == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("OFFLOAD=none requires a CUDA device")
        pipe.to("cuda")

    logger.info("Qwen-Image-2.1 ready (offload=%s)", mode)
    return QwenImage21Runtime(pipe)
