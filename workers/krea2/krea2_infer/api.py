"""Model-specific inference adapter shared by image entrypoints."""

import logging

from shared.api import GenerationResult

logger = logging.getLogger("krea2.api")


def infer(pipe, norm, seed, raw_loras):
    loras = pipe.loras.normalize(raw_loras)
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
        images = pipe.edit(**kwargs)
    else:
        images = pipe.generate(**kwargs, num_images=norm.num_images)
    options = {"mu": effective_mu, "grounding_px": norm.grounding_px,
               "ref_boost": norm.ref_boost, "ref_boost_a": norm.ref_boost_a,
               "ref_position": norm.ref_position, "ref_position_a": norm.ref_position_a,
               "fit_mode": norm.fit_mode}
    legacy = {name: options[name] for name in ("grounding_px", "ref_boost", "fit_mode")} if norm.type == "image_edit" else {}
    return GenerationResult(images, loras, options, legacy)
