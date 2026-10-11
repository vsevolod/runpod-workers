"""RunPod serverless handler — Qwen-Image-2.1 (Diffusers, no ComfyUI).

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

from qwen_image_21_infer.diagnostics import log_stage, startup_diagnostics
from qwen_image_21_infer.lora import LoRAError
from qwen_image_21_infer.pipeline import load_pipeline
from qwen_image_21_infer.request import RequestError, normalize_job_input
from qwen_image_21_infer.runtime_config import DEFAULT_MODEL_DIR
from schemas import INPUT_SCHEMA

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("qwen_image_21.handler")


class ModelHandler:
    """Load models once at worker start (FlashBoot-friendly, predictable cold start)."""

    def __init__(self):
        self.pipe = None
        self.load_models()

    def load_models(self):
        model_dir = os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)
        logger.info("Initializing Qwen-Image-2.1-Turbo from MODEL_DIR=%s", model_dir)
        with startup_diagnostics():
            self.pipe = load_pipeline(model_dir=model_dir)


MODELS = ModelHandler()


def _save_and_upload_images(images, job_id: str) -> list[str]:
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls: list[str] = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path, format="PNG")

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
    """Fallback encoder without temp files (used if job id missing or upload fails)."""
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
        loras = MODELS.pipe.loras.normalize(validated_input["loras"])
    except (RequestError, LoRAError) as err:
        return {"error": str(err)}

    seed = norm.seed
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")

    size_label = (
        f"{norm.width}x{norm.height}"
        if norm.width is not None
        else f"from-source@{norm.output_resolution}"
    )
    logger.info(
        "job type=%s steps=%s guidance=%s size=%s refs=%s images_out=%s loras=%s",
        norm.type,
        norm.num_inference_steps,
        norm.guidance_scale,
        size_label,
        len(norm.images),
        norm.num_images,
        ",".join(f"{item.name}@{item.strength:g}" for item in loras) or "none",
    )

    try:
        with log_stage("inference"):
            images = MODELS.pipe.run(norm, seed=int(seed), loras=loras)
        out_w, out_h = images[0].size
    except LoRAError as err:
        logger.warning("Invalid LoRA request: %s", err)
        return {"error": str(err)}
    except torch.cuda.OutOfMemoryError:
        logger.exception("CUDA OOM during generation")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "error": (
                "CUDA out of memory. On 24 GB keep OFFLOAD=cpu and start at "
                "1024. Native 2048 needs a larger GPU or OFFLOAD=sequential. "
                "Edit encodes every reference plus the target."
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
        with log_stage("image_delivery"):
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
        "num_inference_steps": norm.num_inference_steps,
        "guidance_scale": norm.guidance_scale,
        "output_resolution": norm.output_resolution,
        "loras": [item.as_dict() for item in loras],
    }
    if norm.type == "image_edit":
        payload["num_refs"] = len(norm.images)
    return payload


runpod.serverless.start({"handler": generate_image})
