"""RunPod serverless handler — Krea 2 Turbo FP8 / INT8 ConvRot (thin worker, no ComfyUI).

Portions are adapted from runpod-workers/worker-sdxl under the MIT License;
see LICENSES/RUNPOD-WORKER-SDXL-MIT.txt.
"""

from __future__ import annotations

import base64
import logging
import os
import traceback
from io import BytesIO

import runpod
import torch
from runpod.serverless.utils import rp_cleanup, rp_upload
from runpod.serverless.utils.rp_validator import validate

from krea2_infer import load_pipeline
from krea2_infer.lora import LoRAError
from krea2_infer.request import RequestError, normalize_job_input
from schemas import INPUT_SCHEMA

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


def _save_and_upload_images(images, job_id: str) -> list[str]:
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls: list[str] = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        if os.environ.get("BUCKET_ENDPOINT_URL"):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


def _images_to_base64_data_urls(images) -> list[str]:
    """Fallback encoder without temp files (used if job id missing)."""
    urls: list[str] = []
    for image in images:
        buf = BytesIO()
        image.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode("utf-8")
        urls.append(f"data:image/png;base64,{data}")
    return urls


@torch.inference_mode()
def generate_image(job: dict):
    """Generate or edit image(s). RunPod handler entrypoint."""
    try:
        job_input = job["input"]
    except (KeyError, TypeError):
        return {"error": "Job must contain an 'input' object"}

    raw_keys = set(job_input.keys())
    validated = validate(job_input, INPUT_SCHEMA)
    if "errors" in validated:
        return {"error": validated["errors"]}
    validated_input = validated["validated_input"]

    try:
        norm = normalize_job_input(validated_input, raw_keys=raw_keys)
    except RequestError as err:
        return {"error": str(err)}

    seed = norm.seed
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")

    try:
        loras = MODELS.pipe.loras.normalize(validated_input["loras"])
        if loras:
            logger.info(
                "LoRA request: %s",
                ", ".join(f"{item.name}@{item.strength:g}" for item in loras),
            )

        if norm.type == "image_generate":
            images = MODELS.pipe.generate(
                prompt=norm.prompt,
                width=int(norm.width),
                height=int(norm.height),
                steps=int(norm.num_inference_steps),
                guidance=float(norm.guidance_scale),
                seed=int(seed),
                mu=float(norm.mu) if norm.mu is not None else None,
                num_images=int(norm.num_images),
                negative_prompt=norm.negative_prompt,
                loras=loras,
            )
            out_w, out_h = int(norm.width), int(norm.height)
        elif norm.type == "image_edit":
            if not loras:
                logger.warning(
                    "image_edit without loras; identity edit quality will be poor"
                )
            # edit pins mu: None → 1.15 inside sample_edit
            edit_mu = float(norm.mu) if norm.mu is not None else 1.15
            images = MODELS.pipe.edit(
                prompt=norm.prompt,
                sources=norm.images,
                width=norm.width,
                height=norm.height,
                steps=int(norm.num_inference_steps),
                guidance=float(norm.guidance_scale),
                seed=int(seed),
                mu=edit_mu,
                negative_prompt=norm.negative_prompt,
                grounding_px=int(norm.grounding_px),
                ref_boost=float(norm.ref_boost),
                fit_mode=str(norm.fit_mode),
                loras=loras,
            )
            out_w, out_h = images[0].size  # PIL (W, H)
        else:
            return {"error": f"unsupported type: {norm.type}"}
    except LoRAError as err:
        logger.warning("Invalid LoRA request: %s", err)
        return {"error": str(err)}
    except torch.cuda.OutOfMemoryError:
        logger.exception("CUDA OOM during generation")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "error": (
                "CUDA out of memory. Need ~24 GB with TE offload after encode, "
                "or a larger GPU / lower resolution. Edit uses ~2× image tokens "
                "per ref (+target); two refs ≈ ~3× image tokens."
            ),
            "refresh_worker": True,
        }
    except FileNotFoundError as err:
        logger.exception("Missing model file")
        return {"error": str(err)}
    except Exception as err:
        logger.exception("Generation failed")
        return {
            "error": f"{type(err).__name__}: {err}",
            "traceback": traceback.format_exc(),
            "refresh_worker": True,
        }

    job_id = job.get("id") or "local"
    try:
        image_urls = _save_and_upload_images(images, str(job_id))
    except Exception:
        logger.exception("Upload/save failed; falling back to in-memory base64")
        image_urls = _images_to_base64_data_urls(images)

    payload = {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": int(seed),
        "width": out_w,
        "height": out_h,
        "type": norm.type,
        "loras": [selection.as_dict() for selection in loras],
    }
    if norm.type == "image_edit":
        payload.update(
            {
                "grounding_px": norm.grounding_px,
                "ref_boost": norm.ref_boost,
                "fit_mode": norm.fit_mode,
                "num_refs": len(norm.images),
            }
        )
    return payload


runpod.serverless.start({"handler": generate_image})
