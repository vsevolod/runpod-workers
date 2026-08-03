"""RunPod serverless handler — CivitAI LoRA downloader (CPU, volume write)."""

from __future__ import annotations

import logging
import os

import runpod
from runpod.serverless.utils.rp_validator import validate

from download import get_civitai_token, resolve_lora_dir, run_batch
from schemas import INPUT_SCHEMA

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("lora_downloader.handler")


def handler(job: dict) -> dict:
    if "input" not in job or not isinstance(job["input"], dict):
        return {"error": "job must contain an input object"}

    validated = validate(job["input"], INPUT_SCHEMA)
    if "errors" in validated:
        return {"error": str(validated["errors"])}

    payload = validated["validated_input"]
    items = payload["items"]

    try:
        token = get_civitai_token()
        lora_dir = resolve_lora_dir()
    except (ValueError, OSError) as err:
        return {"error": str(err)}

    # Flat payload — RunPod wraps as output.
    return run_batch(items, token=token, lora_dir=lora_dir)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
