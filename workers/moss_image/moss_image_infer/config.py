"""Immutable upstream assets; checkpoints live on container disk."""

import os
from pathlib import Path

MODEL_ID = 'OpenMOSS-Team/MOSS-Image-SFT'
MODEL_REVISION = '2c7954437269461767f533b006823e8f85973a60'
LORA_ID = 'lodestones/MOSS-Image-SFT-8-steps-dmd'
LORA_REVISION = 'f95df83b5e90ae91ea412434b3f9b7f1c8c6314f'
LORA_FILENAME = 'lora_latest.safetensors'
LORA_STRENGTH = .7
EDIT_ID = 'chinoll/Moss-Image-Edit'
EDIT_REVISION = '3be9c40037e1b89b17aa85ff6f8ee805872aa558'
EDIT_FILES = ('adapter_config.json', 'adapter_model.safetensors')
EDIT_CONTRACT = 'ref-prefix-only-t0__target-text-original-moss-v1'


def model_directory():
    return Path(os.environ.get('MOSS_MODEL_DIR', '/models/moss_image_sft'))


def offload_mode():
    mode = os.environ.get('MOSS_OFFLOAD', 'cpu').strip().lower()
    if mode not in ('cpu', 'none'):
        raise ValueError('MOSS_OFFLOAD must be cpu or none')
    return mode
