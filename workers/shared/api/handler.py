"""RunPod image request lifecycle, independent of each worker's ML runtime.

Portions adapted from runpod-workers/worker-sdxl; see
LICENSES/RUNPOD-WORKER-SDXL-MIT.txt.
"""

from contextlib import nullcontext
from dataclasses import dataclass, field
import os
import traceback

from .contract import RequestError, parse_job_input
from .images import images_to_base64_data_urls, save_and_upload_images


@dataclass
class GenerationResult:
    images: list
    loras: list
    model_options: dict
    metadata: dict = field(default_factory=dict)


def resolve_seed(seed):
    return int.from_bytes(os.urandom(4), "big") if seed is None else seed


def handle_job(job, *, profile, normalize, infer, logger, log_stage=None,
               deliver=save_and_upload_images, client_errors=(), oom_errors=(),
               oom_cleanup=None, oom_message="CUDA out of memory"):
    stage = log_stage or (lambda *args, **kwargs: nullcontext())
    try:
        parsed = parse_job_input(job, profile)
        request = normalize(parsed.values, raw_keys=parsed.raw_keys)
    except RequestError as err:
        return {"error": str(err)}
    seed = resolve_seed(request.seed)
    logger.info(
        "job id=%s model=%s type=%s steps=%s guidance=%s sampler=%s seed=%s refs=%s size=%sx%s model_options=%s",
        job.get("id"), profile.name, request.type, request.num_inference_steps,
        request.guidance_scale, request.sampler, seed, len(request.images),
        getattr(request, "width", "auto"), getattr(request, "height", "auto"), parsed.values["model_options"],
    )
    logger.info("reference_megapixels requested=%s resolved=%s vram_gib=%s",
                request.requested_reference_megapixels, request.reference_megapixels, request.vram_gib)
    try:
        with stage("inference"):
            generated = infer(request, seed, parsed.values["loras"])
            if not generated.images:
                raise RuntimeError("Pipeline returned no images")
            width, height = generated.images[0].size
    except (RequestError, *client_errors) as err:
        logger.warning("Invalid generation request: %s", err)
        return {"error": str(err)}
    except oom_errors:
        logger.exception("CUDA OOM during generation")
        if oom_cleanup is not None:
            oom_cleanup()
        return {"error": oom_message, "refresh_worker": True}
    except FileNotFoundError as err:
        logger.exception("Missing model file")
        return {"error": str(err)}
    except Exception as err:
        logger.exception("Generation failed")
        return {"error": f"{type(err).__name__}: {err}", "traceback": traceback.format_exc(), "refresh_worker": True}
    try:
        with stage("image_delivery"):
            urls = deliver(generated.images, str(job.get("id") or "local"))
    except Exception:
        logger.exception("Upload/save failed; falling back to in-memory base64")
        urls = images_to_base64_data_urls(generated.images)
    payload = {
        **generated.metadata,
        "images": urls, "seed": seed, "width": width, "height": height,
        "type": request.type, "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale, "sampler": request.sampler,
        "model_options": generated.model_options,
        "reference_megapixels": request.requested_reference_megapixels,
        "resolved_reference_megapixels": request.reference_megapixels,
        "vram_gib": request.vram_gib,
        "loras": [item.as_dict() for item in generated.loras],
    }
    if request.type == "image_edit":
        payload["num_refs"] = len(request.images)
    return payload
