"""RunPod serverless handler — MiniMax H3 T2V (thin worker, no ComfyUI).

Portions of the RunPod handler pattern are adapted from runpod-workers/worker-sdxl
under the MIT License; see LICENSES/RUNPOD-WORKER-SDXL-MIT.txt.
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import traceback
from pathlib import Path

from h3_infer.delivery import (
    MAX_INLINE_VIDEO_BYTES_DEFAULT,
    bucket_configured,
    choose_delivery,
    require_bucket_or_exit,
)
from h3_infer.request import RequestError, normalize_t2v_input
from schemas import INPUT_SCHEMA

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("minimax_h3.handler")


def _max_inline() -> int:
    raw = os.environ.get("MAX_INLINE_VIDEO_BYTES")
    if raw is None or raw == "":
        return MAX_INLINE_VIDEO_BYTES_DEFAULT
    try:
        return int(raw)
    except ValueError:
        return MAX_INLINE_VIDEO_BYTES_DEFAULT


def _validate(job_input: dict, schema: dict) -> dict:
    """Indirection so unit tests can mock without importing runpod."""
    from runpod.serverless.utils.rp_validator import validate

    return validate(job_input, schema)


def _upload_file_to_bucket(**kwargs):
    """Indirection so unit tests can mock without importing runpod."""
    from runpod.serverless.utils import rp_upload

    return rp_upload.upload_file_to_bucket(**kwargs)


def _meta(req) -> dict:
    return {
        "width": req.width,
        "height": req.height,
        "length": req.length,
        "seed": req.seed,
        "requested_duration": req.requested_duration,
        "output_duration": req.output_duration,
        "num_inference_steps": 50,
    }


class ModelHandler:
    """Load MiniMax H3 once at worker start (FlashBoot-friendly)."""

    def __init__(self) -> None:
        from h3_infer.pipeline import H3Pipeline

        require_bucket_or_exit()
        model_dir = os.environ.get("MODEL_DIR", "/runpod-volume/minimax_h3")
        logger.info("Initializing H3Pipeline from MODEL_DIR=%s", model_dir)
        self.pipe = H3Pipeline(model_dir=model_dir)
        logger.info("H3Pipeline ready")


_MODELS: ModelHandler | None = None


def get_models() -> ModelHandler:
    """Lazy singleton; forced at ``__main__`` for serverless warm path."""
    global _MODELS
    if _MODELS is None:
        _MODELS = ModelHandler()
    return _MODELS


def handler(job: dict) -> dict:
    """Generate T2V MP4 and deliver via bucket URL or base64."""
    try:
        job_input = job["input"]
    except (KeyError, TypeError):
        return {"error": "Job must contain an 'input' object"}

    try:
        validated = _validate(job_input, INPUT_SCHEMA)
    except Exception as err:  # pragma: no cover - validator import/runtime
        logger.exception("validate failed")
        return {"error": f"validation error: {err}"}

    if "errors" in validated:
        return {"error": validated["errors"]}

    validated_input = validated["validated_input"]
    # seed == -1 means "missing" / random (schema default).
    if validated_input.get("seed") == -1:
        validated_input = {**validated_input, "seed": -1}

    try:
        req = normalize_t2v_input(validated_input)
    except RequestError as err:
        return {"error": str(err)}

    job_id = str(job.get("id") or "local")
    # Prefer /{job_id} on RunPod; fall back to /tmp when not writable (unit tests).
    work_dir = Path(f"/{job_id}")
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        work_dir = Path("/tmp/minimax_h3_jobs") / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = work_dir / "output.mp4"

    try:
        models = get_models()
        models.pipe.generate_t2v(req, mp4_path)

        raw_size = mp4_path.stat().st_size
        plan = choose_delivery(bucket_configured(), raw_size, _max_inline())
        meta = _meta(req)

        if plan.mode == "error":
            return {"error": plan.error, **meta}

        if plan.mode == "base64":
            raw = mp4_path.read_bytes()
            b64 = base64.b64encode(raw).decode("ascii")
            return {
                **meta,
                "video_base64": b64,
                "content_type": "video/mp4",
                "size_bytes": raw_size,
            }

        # url mode
        url = _upload_file_to_bucket(
            file_name=f"{job_id}/output.mp4",
            file_location=str(mp4_path),
            bucket_name=os.environ["BUCKET_NAME"],
            extra_args={"ContentType": "video/mp4"},
        )
        if not str(url).startswith(("http://", "https://")):
            return {
                "error": "bucket upload returned non-URL path; check BUCKET_* credentials",
                **meta,
            }
        return {**meta, "video_url": url, "size_bytes": raw_size}

    except Exception as err:
        # Prefer torch.cuda.OutOfMemoryError when torch is available.
        oom = False
        try:
            import torch

            if isinstance(err, torch.cuda.OutOfMemoryError):
                oom = True
        except Exception:
            pass
        if type(err).__name__ == "OutOfMemoryError":
            oom = True

        logger.error("job failed: %s\n%s", err, traceback.format_exc())
        if oom:
            return {"error": "CUDA out of memory", "refresh_worker": True}
        return {
            "error": f"{type(err).__name__}: {err}",
            "refresh_worker": True,
        }
    finally:
        try:
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    import runpod

    # Force load at process start for serverless warm path.
    get_models()
    runpod.serverless.start({"handler": handler})
