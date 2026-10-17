"""RunPod serverless handler — Qwen-Image-2.1 (Diffusers, no ComfyUI).

Portions are adapted from runpod-workers/worker-sdxl under the MIT License;
see LICENSES/RUNPOD-WORKER-SDXL-MIT.txt.
"""

from __future__ import annotations

import logging
import os

import runpod
import torch

from qwen_image_21_infer.diagnostics import log_stage
from qwen_image_21_infer.lora import LoRAError
from qwen_image_21_infer.model_manager import ModelManager
from qwen_image_21_infer.request import normalize_job_input
from qwen_image_21_infer.runtime_config import resolve_model_id
from shared.api import GenerationResult, QWEN_API, handle_job
from shared.api.images import save_and_upload_images as _save_and_upload_images

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("qwen_image_21.handler")


MODELS = ModelManager()


def _infer(norm, seed, raw_loras):
    loras = MODELS.normalize_loras(raw_loras)
    logger.info("LoRA request: %s", ",".join(f"{item.name}@{item.strength:g}" for item in loras) or "none")
    images = MODELS.run(norm, seed=seed, loras=loras)
    return GenerationResult(
        images, loras,
        {"variant": norm.model_variant, "output_resolution": norm.requested_output_resolution},
        {"model_variant": norm.model_variant, "model_id": resolve_model_id(norm.model_variant),
         "scheduler": norm.scheduler, "output_resolution": norm.output_resolution},
    )


def _oom_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def generate_image(job: dict):
    return handle_job(
        job, profile=QWEN_API, normalize=normalize_job_input, infer=_infer,
        logger=logger, log_stage=log_stage, deliver=_save_and_upload_images,
        client_errors=(LoRAError,), oom_errors=(torch.cuda.OutOfMemoryError,),
        oom_cleanup=_oom_cleanup,
        oom_message=(
            "CUDA out of memory. On 24 GB keep OFFLOAD=cpu and start at "
            "1024. Native 2048 needs a larger GPU or OFFLOAD=sequential. "
            "Edit encodes every reference plus the target."
        ),
    )


runpod.serverless.start({"handler": generate_image})
