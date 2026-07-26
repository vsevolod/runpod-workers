"""RunPod serverless handler — JoyCaption Beta One image captioning.

Portions of the RunPod handler pattern are adapted from runpod-workers/worker-sdxl
under the MIT License; see LICENSES/RUNPOD-WORKER-SDXL-MIT.txt.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import runpod
import torch
from runpod.serverless.utils.rp_validator import validate
from transformers import AutoProcessor, LlavaForConditionalGeneration

from caption_utils import (
    DEFAULT_MODEL_ID,
    SYSTEM_MESSAGE,
    ImageError,
    decode_image_base64,
    parse_local_files_only,
    parse_max_image_pixels,
    resolve_prompt,
)
from schemas import INPUT_SCHEMA

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("joycaption.handler")


def _model_source() -> str:
    """Prefer local MODEL_DIR when it looks like a snapshot; else HF MODEL_ID."""
    model_dir = os.environ.get("MODEL_DIR", "/runpod-volume/joycaption")
    model_id = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
    path = Path(model_dir)
    if path.is_dir() and (path / "config.json").is_file():
        return str(path)
    return model_id


class ModelHandler:
    """Load JoyCaption once at worker start (FlashBoot-friendly)."""

    def __init__(self) -> None:
        self.processor = None
        self.model = None
        self.model_id = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
        self.source = _model_source()
        self.load_models()

    def load_models(self) -> None:
        local_files_only = parse_local_files_only()
        logger.info(
            "Loading JoyCaption from %s (local_files_only=%s)",
            self.source,
            local_files_only,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.source,
            local_files_only=local_files_only,
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.source,
            torch_dtype=torch.bfloat16,
            device_map=0,
            local_files_only=local_files_only,
        )
        self.model.eval()
        logger.info("JoyCaption ready")


# Load at import / process start (not lazy on first request).
MODELS = ModelHandler()


@torch.inference_mode()
def caption_image(
    image,
    prompt: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> str:
    """Run official JoyCaption generate path; return stripped caption text."""
    processor = MODELS.processor
    model = MODELS.model

    convo = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]
    # HF Llava chat handling is fragile — keep this exact combination
    # (see fancyfeast / fpgaminer JoyCaption README).
    convo_string = processor.apply_chat_template(
        convo, tokenize=False, add_generation_prompt=True
    )
    if not isinstance(convo_string, str):
        raise RuntimeError("apply_chat_template did not return a string")

    inputs = processor(
        text=[convo_string],
        images=[image],
        return_tensors="pt",
    ).to("cuda")
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
        suppress_tokens=None,
        use_cache=True,
        temperature=float(temperature),
        top_k=None,
        top_p=float(top_p),
    )[0]

    generate_ids = generate_ids[inputs["input_ids"].shape[1] :]
    caption = processor.tokenizer.decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return caption.strip()


def handler(job: dict):
    """Caption a single base64 image. RunPod handler entrypoint."""
    try:
        job_input = job["input"]
    except (KeyError, TypeError):
        return {"error": "Job must contain an 'input' object"}

    validated = validate(job_input, INPUT_SCHEMA)
    if "errors" in validated:
        return {"error": validated["errors"]}
    data = validated["validated_input"]

    try:
        max_pixels = parse_max_image_pixels()
    except ValueError as err:
        return {"error": str(err)}

    try:
        image = decode_image_base64(data["image"], max_pixels=max_pixels)
    except ImageError as err:
        return {"error": str(err)}

    prompt = resolve_prompt(data.get("prompt"))
    max_new_tokens = int(data["max_new_tokens"])
    temperature = float(data["temperature"])
    top_p = float(data["top_p"])

    try:
        caption = caption_image(
            image,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    except torch.cuda.OutOfMemoryError:
        logger.exception("CUDA OOM during captioning")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "error": (
                "CUDA out of memory. JoyCaption bf16 needs ~17 GB VRAM; "
                "use a 24 GB class GPU."
            ),
            "refresh_worker": True,
        }
    except Exception as err:
        logger.exception("Captioning failed")
        # Spec: no stack traces in client payload — logs only.
        return {
            "error": f"{type(err).__name__}: {err}",
            "refresh_worker": True,
        }

    return {
        "caption": caption,
        "prompt": prompt,
        "model": MODELS.model_id,
    }


runpod.serverless.start({"handler": handler})
