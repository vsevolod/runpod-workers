"""RunPod serverless handler — Krea 2 Turbo FP8 (thin worker, no ComfyUI).

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
        logger.info("Initializing Krea2 pipeline from MODEL_DIR=%s", model_dir)
        self.pipe = load_pipeline(model_dir=model_dir)
        logger.info("Krea2 pipeline ready")


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
    """Generate image(s) from text prompt. RunPod handler entrypoint."""
    try:
        job_input = job["input"]
    except (KeyError, TypeError):
        return {"error": "Job must contain an 'input' object"}

    validated = validate(job_input, INPUT_SCHEMA)
    if "errors" in validated:
        return {"error": validated["errors"]}
    job_input = validated["validated_input"]

    prompt = job_input["prompt"]
    if not prompt or not str(prompt).strip():
        return {"error": "prompt must be a non-empty string"}

    seed = job_input["seed"]
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")

    width = int(job_input["width"])
    height = int(job_input["height"])
    # Latent grid must be multiple of 16 (ae.compression 8 * patch 2).
    # Official sampler pads up; we still reject misaligned sizes for predictable output.
    if width % 16 != 0 or height % 16 != 0:
        return {
            "error": (
                f"width and height must be multiples of 16 (got {width}x{height})"
            )
        }

    try:
        images = MODELS.pipe.generate(
            prompt=str(prompt),
            width=width,
            height=height,
            steps=int(job_input["num_inference_steps"]),
            guidance=float(job_input["guidance_scale"]),
            seed=int(seed),
            mu=float(job_input["mu"]) if job_input["mu"] is not None else None,
            num_images=int(job_input["num_images"]),
            negative_prompt=job_input.get("negative_prompt"),
        )
    except torch.cuda.OutOfMemoryError:
        logger.exception("CUDA OOM during generation")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "error": "CUDA out of memory. Use a 24 GB+ GPU or lower resolution.",
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

    return {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": int(seed),
        "width": width,
        "height": height,
    }


runpod.serverless.start({"handler": generate_image})
