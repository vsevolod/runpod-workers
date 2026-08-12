"""RunPod serverless handler — MiniMax H3 via headless native ComfyUI.

Architecture: start.sh launches pinned ComfyUI on 127.0.0.1:8188, then this
handler. Product input → inject frozen API workflow → POST /prompt → poll
history → path from SaveVideo node only → S3 URL or limited base64.
"""

from __future__ import annotations

import base64
import logging
import os
import random
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import requests
import runpod
from runpod.serverless.utils import rp_cleanup, rp_upload

from image_input import ImageInputError, cleanup_staged, stage_image
from workflow import (
    DEFAULT_DURATION,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    SAVE_VIDEO_NODE,
    inject_product,
    load_workflow,
    validate_canvas,
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("minimax_h3_comfy.handler")

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_OUTPUT_DIR = Path(
    os.environ.get("COMFY_OUTPUT_DIR", "/comfyui/output")
)
COMFY_INPUT_DIR = Path(os.environ.get("COMFY_INPUT_DIR", "/comfyui/input"))
MAX_INLINE_VIDEO_BYTES = int(os.environ.get("MAX_INLINE_VIDEO_BYTES", str(7_000_000)))
PROMPT_TIMEOUT_S = float(os.environ.get("COMFY_PROMPT_TIMEOUT_S", "3600"))
POLL_INTERVAL_S = float(os.environ.get("COMFY_POLL_INTERVAL_S", "1.0"))
MODEL_ID = "minimax_h3_fl2va_pruned_int8_convrot"

BUCKET_KEYS = (
    "BUCKET_ENDPOINT_URL",
    "BUCKET_ACCESS_KEY_ID",
    "BUCKET_SECRET_ACCESS_KEY",
    "BUCKET_NAME",
)


def bucket_state(env: dict[str, str] | None = None) -> str:
    """Return 'none' | 'full' | 'partial' for BUCKET_* configuration."""
    src = env if env is not None else os.environ
    present = [k for k in BUCKET_KEYS if (src.get(k) or "").strip()]
    if len(present) == 0:
        return "none"
    if len(present) == 4:
        return "full"
    return "partial"


def require_bucket_config_or_exit() -> str:
    """Startup: partial BUCKET_* is misconfiguration — exit process."""
    present = [k for k in BUCKET_KEYS if (os.environ.get(k) or "").strip()]
    missing = [k for k in BUCKET_KEYS if k not in present]
    # Log names only (never secret values)
    logger.info(
        "BUCKET_* set=%s missing=%s",
        present,
        missing,
    )
    state = bucket_state()
    if state == "partial":
        logger.error(
            "Partial BUCKET_* config: present=%s missing=%s. "
            "Set all four for URL delivery, or none for inline base64.",
            present,
            missing,
        )
        raise SystemExit(2)
    logger.info("Delivery mode: %s", "S3 URL" if state == "full" else "inline base64")
    return state


def path_from_savevideo(history_entry: dict[str, Any], output_dir: Path) -> Path:
    """Resolve MP4 path from SaveVideo history only (PINS.md contract)."""
    outputs = history_entry.get("outputs") or {}
    node_out = outputs.get(SAVE_VIDEO_NODE)
    if not isinstance(node_out, dict):
        raise RuntimeError(
            f"SaveVideo node {SAVE_VIDEO_NODE!r} missing from history outputs; "
            f"keys={list(outputs.keys())}"
        )
    images = node_out.get("images")
    if not images:
        raise RuntimeError(
            f"SaveVideo node {SAVE_VIDEO_NODE!r} has no 'images' entries "
            f"(got keys={list(node_out.keys())}); refusing glob fallback"
        )
    meta = images[0]
    filename = meta.get("filename")
    subfolder = meta.get("subfolder") or ""
    ftype = meta.get("type") or "output"
    if not filename:
        raise RuntimeError(f"SaveVideo metadata missing filename: {meta!r}")
    if ftype != "output":
        # Still resolve under output_dir when type is output; temp would differ.
        logger.warning("SaveVideo type=%r (expected 'output')", ftype)
    path = output_dir / subfolder / filename if subfolder else output_dir / filename
    if not path.is_file():
        raise RuntimeError(f"SaveVideo file not found on disk: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"SaveVideo file is empty: {path}")
    return path


def wait_for_comfy(timeout_s: float = 300.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{COMFY_URL}/system_stats", timeout=5)
            if r.status_code == 200:
                logger.info("ComfyUI ready at %s", COMFY_URL)
                return
        except Exception as e:  # noqa: BLE001 — poll until up
            last_err = e
        time.sleep(1.0)
    raise RuntimeError(f"ComfyUI not ready at {COMFY_URL}: {last_err}")


def submit_prompt(workflow: dict[str, Any]) -> str:
    client_id = str(uuid.uuid4())
    r = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=60,
    )
    r.raise_for_status()
    body = r.json()
    if body.get("error"):
        raise RuntimeError(f"Comfy /prompt error: {body['error']}")
    if body.get("node_errors"):
        raise RuntimeError(f"Comfy node_errors: {body['node_errors']}")
    prompt_id = body.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"Comfy /prompt missing prompt_id: {body}")
    return prompt_id


def poll_history(prompt_id: str) -> dict[str, Any]:
    deadline = time.time() + PROMPT_TIMEOUT_S
    while time.time() < deadline:
        r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=60)
        r.raise_for_status()
        data = r.json()
        if prompt_id in data:
            entry = data[prompt_id]
            status = (entry.get("status") or {}).get("status_str")
            if status == "error":
                raise RuntimeError(f"Comfy job error: {entry.get('status')}")
            # outputs appear when finished
            if entry.get("outputs"):
                return entry
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"Comfy history timeout after {PROMPT_TIMEOUT_S}s for {prompt_id}")


def _optional_image_field(job_input: dict[str, Any], key: str) -> str | None:
    if key not in job_input or job_input[key] is None:
        return None
    val = job_input[key]
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"{key} must be a non-empty string when provided")
    return val.strip()


def normalize_input(job_input: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(job_input, dict):
        raise ValueError("input must be an object")
    prompt = job_input.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt is required (non-empty string)")

    width = int(job_input.get("width", DEFAULT_WIDTH))
    height = int(job_input.get("height", DEFAULT_HEIGHT))
    validate_canvas(width, height)

    duration = float(job_input.get("duration", DEFAULT_DURATION))
    if duration <= 0:
        raise ValueError("duration must be positive")

    seed_raw = job_input.get("seed", -1)
    seed = int(seed_raw)
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    first_image = _optional_image_field(job_input, "first_image")
    last_image = _optional_image_field(job_input, "last_image")
    if last_image and not first_image:
        raise ValueError("last_image requires first_image")

    return {
        "prompt": prompt.strip(),
        "width": width,
        "height": height,
        "duration": duration,
        "seed": seed,
        "first_image": first_image,
        "last_image": last_image,
    }


def deliver_video(mp4: Path, job_id: str) -> dict[str, Any]:
    state = bucket_state()
    size = mp4.stat().st_size
    if state == "full":
        # rp_upload.upload_image is image-oriented; use file upload helper if present
        url = _upload_video(job_id, mp4)
        return {"video_url": url, "delivery": "url", "bytes": size}
    if size > MAX_INLINE_VIDEO_BYTES:
        raise RuntimeError(
            f"Video is {size} bytes > MAX_INLINE_VIDEO_BYTES={MAX_INLINE_VIDEO_BYTES}; "
            "configure full BUCKET_* for URL delivery"
        )
    b64 = base64.b64encode(mp4.read_bytes()).decode("ascii")
    return {
        "video": f"data:video/mp4;base64,{b64}",
        "delivery": "base64",
        "bytes": size,
    }


def _upload_video(job_id: str, path: Path) -> str:
    """Upload via runpod rp_upload (S3-compatible when BUCKET_* set)."""
    # Prefer generic file upload; fall back to upload_image path API.
    if hasattr(rp_upload, "upload_file_to_bucket"):
        return rp_upload.upload_file_to_bucket(
            file_name=path.name,
            file_location=str(path),
            prefix=job_id,
        )
    # Older runpod: upload_image still works for arbitrary files in practice
    return rp_upload.upload_image(job_id, str(path))


def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_id = str(job.get("id") or uuid.uuid4())
    staged_first: str | None = None
    staged_last: str | None = None
    try:
        params = normalize_input(job.get("input") or {})
        if params["first_image"]:
            try:
                staged_first = stage_image(
                    params["first_image"],
                    input_dir=COMFY_INPUT_DIR,
                    basename_stem=f"{job_id}_first",
                )
            except ImageInputError as err:
                raise ValueError(str(err)) from err
        if params["last_image"]:
            try:
                staged_last = stage_image(
                    params["last_image"],
                    input_dir=COMFY_INPUT_DIR,
                    basename_stem=f"{job_id}_last",
                )
            except ImageInputError as err:
                raise ValueError(str(err)) from err

        wf = inject_product(
            load_workflow(),
            prompt=params["prompt"],
            width=params["width"],
            height=params["height"],
            duration=params["duration"],
            seed=params["seed"],
            first_image_name=staged_first,
            last_image_name=staged_last,
        )
        prompt_id = submit_prompt(wf)
        logger.info(
            "Submitted prompt_id=%s job_id=%s i2v=%s fl2v=%s",
            prompt_id,
            job_id,
            bool(staged_first),
            bool(staged_last),
        )
        history = poll_history(prompt_id)
        mp4 = path_from_savevideo(history, COMFY_OUTPUT_DIR)
        logger.info("SaveVideo path=%s size=%s", mp4, mp4.stat().st_size)
        delivery = deliver_video(mp4, job_id)
        # best-effort cleanup of job staging only (not Comfy output tree wholesale)
        try:
            rp_cleanup.clean([f"/{job_id}"])
        except Exception:  # noqa: BLE001
            pass
        mode = "t2v"
        if staged_first and staged_last:
            mode = "fl2v"
        elif staged_first:
            mode = "i2v"
        return {
            **delivery,
            "width": params["width"],
            "height": params["height"],
            "seed": params["seed"],
            "duration": params["duration"],
            "model": MODEL_ID,
            "mode": mode,
            "prompt_id": prompt_id,
            "filename": mp4.name,
        }
    except Exception as e:  # noqa: BLE001 — surface to RunPod
        logger.error("job failed: %s\n%s", e, traceback.format_exc())
        return {"error": str(e)}
    finally:
        cleanup_staged(COMFY_INPUT_DIR, staged_first, staged_last)


if __name__ == "__main__":
    logger.info(
        "handler start MODEL_DIR=%s COMFY_URL=%s COMFY_OUTPUT_DIR=%s",
        os.environ.get("MODEL_DIR"),
        COMFY_URL,
        COMFY_OUTPUT_DIR,
    )
    # Fail fast on misconfigured partial BUCKET_* (startup only; keeps unit imports clean).
    require_bucket_config_or_exit()
    # start.sh already waits for Comfy; keep a short re-check by default.
    if os.environ.get("COMFY_WAIT", "1") not in ("0", "false", "False"):
        wait_for_comfy(float(os.environ.get("COMFY_WAIT_TIMEOUT_S", "120")))
    logger.info("entering runpod.serverless.start")
    runpod.serverless.start({"handler": handler})
