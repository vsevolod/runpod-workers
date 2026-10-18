"""RunPod serverless handler — Krea 2 Turbo FP8 / INT8 ConvRot (thin worker, no ComfyUI).

Portions are adapted from runpod-workers/worker-sdxl under the MIT License;
see LICENSES/RUNPOD-WORKER-SDXL-MIT.txt.
"""

from __future__ import annotations

import logging
import os

import runpod
import torch

from krea2_infer import load_pipeline
from krea2_infer.diagnostics import log_stage
from krea2_infer.lora import LoRAError
from krea2_infer.request import normalize_job_input
from shared.api import GenerationResult, KREA_API, handle_job
from shared.api.images import save_and_upload_images as _save_and_upload_images

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("krea2.handler")


class ModelHandler:
    """Load models once at worker start (FlashBoot-friendly, predictable cold start)."""

    def __init__(self):
        self.pipe = None
        self.load_models()

    def load_models(self):
        model_dir = os.environ.get("MODEL_DIR", "/runpod-volume/krea2")
        dit_quant = os.environ.get("DIT_QUANT", "fp8")
        logger.info(
            "Initializing Krea2 pipeline from MODEL_DIR=%s DIT_QUANT=%s",
            model_dir,
            dit_quant,
        )
        self.pipe = load_pipeline(model_dir=model_dir)
        logger.info(
            "Krea2 pipeline ready (quant_mode=%s)",
            getattr(self.pipe, "quant_mode", dit_quant),
        )


# Load at import / process start (not lazy on first request).
MODELS = ModelHandler()


def _infer(norm, seed, raw_loras):
    loras = MODELS.pipe.loras.normalize(raw_loras)
    logger.info("LoRA request: %s", ",".join(f"{item.name}@{item.strength:g}:{item.type}" for item in loras) or "none")
    kwargs = dict(
        prompt=norm.prompt, width=norm.width, height=norm.height,
        steps=norm.num_inference_steps, guidance=norm.guidance_scale,
        seed=seed, mu=norm.mu, negative_prompt=norm.negative_prompt, loras=loras,
    )
    effective_mu = norm.mu
    if norm.type == "image_edit":
        if not loras:
            logger.warning("image_edit without loras; identity edit quality will be poor")
        effective_mu = 1.15 if norm.mu is None else norm.mu
        kwargs.update(sources=norm.images, mu=effective_mu, grounding_px=norm.grounding_px,
                      ref_boost=norm.ref_boost, ref_boost_a=norm.ref_boost_a,
                      ref_position=norm.ref_position, ref_position_a=norm.ref_position_a,
                      fit_mode=norm.fit_mode)
        images = MODELS.pipe.edit(**kwargs)
    else:
        images = MODELS.pipe.generate(**kwargs, num_images=norm.num_images)
    options = {"mu": effective_mu, "grounding_px": norm.grounding_px,
               "ref_boost": norm.ref_boost, "ref_boost_a": norm.ref_boost_a,
               "ref_position": norm.ref_position, "ref_position_a": norm.ref_position_a,
               "fit_mode": norm.fit_mode}
    legacy = {name: options[name] for name in ("grounding_px", "ref_boost", "fit_mode")} if norm.type == "image_edit" else {}
    return GenerationResult(images, loras, options, legacy)


def _oom_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def generate_image(job: dict):
    return handle_job(
        job, profile=KREA_API,
        normalize=lambda values, *, raw_keys: normalize_job_input(
            values, raw_keys=raw_keys, quant_mode=getattr(MODELS.pipe, "quant_mode", "fp8")), infer=_infer,
        logger=logger, log_stage=log_stage, deliver=_save_and_upload_images,
        client_errors=(LoRAError,), oom_errors=(torch.cuda.OutOfMemoryError,),
        oom_cleanup=_oom_cleanup,
        oom_message=(
            "CUDA out of memory. Need ~24 GB with TE offload after encode, "
            "or a larger GPU / lower resolution. Edit uses ~2× image tokens "
            "per ref (+target); two refs ≈ ~3× image tokens."
        ),
    )


runpod.serverless.start({"handler": generate_image})
