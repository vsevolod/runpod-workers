"""Model-specific inference adapter shared by image entrypoints."""

import logging

from shared.api import GenerationResult
from .runtime_config import resolve_model_id

logger = logging.getLogger("qwen_image_21.api")


def infer(manager, norm, seed, raw_loras):
    loras = manager.normalize_loras(raw_loras)
    logger.info("LoRA request: %s", ",".join(f"{item.name}@{item.strength:g}" for item in loras) or "none")
    images = manager.run(norm, seed=seed, loras=loras)
    return GenerationResult(
        images, loras,
        {"variant": norm.model_variant, "output_resolution": norm.requested_output_resolution},
        {"model_variant": norm.model_variant, "model_id": resolve_model_id(norm.model_variant),
         "scheduler": norm.scheduler, "output_resolution": norm.output_resolution},
    )
