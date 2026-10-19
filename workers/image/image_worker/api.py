"""Select a model profile before applying the shared image request lifecycle."""

from contextlib import contextmanager
import logging
import time

import torch

from . import backends  # Establish the model/shared package paths.
from shared.api import handle_job
from shared.api.images import save_and_upload_images

logger = logging.getLogger(__name__)


@contextmanager
def log_stage(name):
    started = time.monotonic()
    logger.info('stage=%s begin', name)
    try:
        yield
    except BaseException:
        logger.info('stage=%s failed elapsed_s=%.2f', name, time.monotonic() - started)
        raise
    else:
        logger.info('stage=%s done elapsed_s=%.2f', name, time.monotonic() - started)


def generate_image(job, manager, *, deliver=save_and_upload_images):
    if not isinstance(job, dict) or not isinstance(job.get('input'), dict):
        return {'error': "Job must contain an 'input' object"}
    model = job['input'].get('model', 'krea2')
    if not isinstance(model, str) or model not in manager.registry:
        return {'error': 'model must be one of: ' + ', '.join(manager.registry)}
    backend = manager.registry[model]
    forwarded = {**job, 'input': {key: value for key, value in job['input'].items() if key != 'model'}}
    output = handle_job(
        forwarded, profile=backend.profile, normalize=backend.normalize,
        infer=lambda norm, seed, loras: manager.infer(model, norm, seed, loras),
        logger=logger, log_stage=log_stage, deliver=deliver,
        client_errors=backend.client_errors, oom_errors=(torch.cuda.OutOfMemoryError,),
        oom_cleanup=manager.close, oom_message=backend.oom_message,
    )
    return {**output, 'model': model}
