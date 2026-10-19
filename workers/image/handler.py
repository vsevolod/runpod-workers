"""One RunPod handler for request-selected image generation models."""

import importlib.metadata
import logging
import os

import runpod
import torch

from image_worker.api import generate_image as dispatch
from image_worker.backends import REGISTRY
from image_worker.manager import ModelManager

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'),
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('image.handler')
logger.info('Image worker ready; models=%s torch=%s cuda=%s transformers=%s diffusers=%s',
            ','.join(REGISTRY), torch.__version__, torch.version.cuda,
            importlib.metadata.version('transformers'), importlib.metadata.version('diffusers'))
MODELS = ModelManager(REGISTRY)


@torch.inference_mode()
def generate_image(job):
    return dispatch(job, MODELS)


runpod.serverless.start({'handler': generate_image})
