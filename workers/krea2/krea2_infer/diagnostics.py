"""Phase timings; optional CUDA synchronization for profiling slow requests."""

import logging
import os
import time
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


@contextmanager
def log_stage(stage, **fields):
    synchronize = (
        os.environ.get("KREA_PROFILE_CUDA", "0") == "1"
        and torch.cuda.is_initialized()
    )
    if synchronize:
        torch.cuda.synchronize()
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    started = time.perf_counter()
    logger.info("stage=%s begin %s cuda_synchronized=%s", stage, details, synchronize)
    try:
        yield
        if synchronize:
            torch.cuda.synchronize()
    except BaseException:
        logger.info("stage=%s failed elapsed_s=%.2f %s", stage, time.perf_counter() - started, details)
        raise
    else:
        logger.info("stage=%s done elapsed_s=%.2f %s", stage, time.perf_counter() - started, details)
