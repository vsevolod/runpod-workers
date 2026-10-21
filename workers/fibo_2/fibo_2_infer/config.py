"""Public base/turbo names map to immutable, separate upstream checkpoints."""

import os
from pathlib import Path

CHECKPOINTS = {
    'base': ('briaai/Fibo-2.0-dev', 'cbf184270ff8f2cf9f2daf464564f37fd34c4a05'),
    'turbo': ('briaai/Fibo-2.0-edit-turbo', '63316ff6d5c4c4ba37729a8a6510bced7e8f585b'),
}


def model_directory(variant):
    return Path(os.environ.get('FIBO_MODEL_DIR') or '/models/fibo_2') / variant


def offload_mode():
    mode = os.environ.get('FIBO_OFFLOAD', 'cpu')
    if mode not in ('cpu', 'none'):
        raise ValueError('FIBO_OFFLOAD must be cpu or none')
    return mode
