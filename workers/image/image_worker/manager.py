"""Serialize inference and release a backend before loading its replacement."""

import gc
import logging
from threading import RLock

logger = logging.getLogger(__name__)


def release_memory():
    gc.collect()
    import torch

    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated()
    return 0


class ModelManager:
    def __init__(self, registry):
        self.registry = dict(registry)
        self._lock = RLock()
        self._backend = None
        self._model = None

    def _unload(self):
        backend, self._backend = self._backend, None
        model, self._model = self._model, None
        if backend is None:
            return
        logger.info('model_unload begin model=%s', model)
        try:
            backend.close()
        finally:
            del backend
            allocated = release_memory()
            logger.info('model_unload done model=%s cuda_allocated_bytes=%s', model, allocated)

    def infer(self, model, norm, seed, raw_loras):
        factory = self.registry.get(model)
        if factory is None:
            raise ValueError(f'Unknown image model: {model}')
        with self._lock:
            if self._model != model:
                self._unload()
                backend = factory()
                logger.info('backend_load begin model=%s', model)
                try:
                    backend.load()
                except BaseException:
                    try:
                        backend.close()
                    except Exception:
                        logger.exception('Failed to close partially loaded model=%s', model)
                    del backend
                    release_memory()
                    raise
                self._backend = backend
                self._model = model
                logger.info('backend_load done model=%s', model)
            return self._backend.infer(norm, seed, raw_loras)

    def close(self):
        with self._lock:
            self._unload()
