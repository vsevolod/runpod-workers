"""Serialize requests and keep only the selected model variant resident."""

import gc
import logging
from threading import RLock

from .diagnostics import log_stage, startup_diagnostics
from .lora import LoRACatalog
from .runtime_config import resolve_lora_directory, resolve_vae_variant

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, *, loader=None, prepare=None):
        if loader is None:
            from .pipeline import load_pipeline
            loader = load_pipeline
        if prepare is None:
            from .model_store import prefetch_model
            prepare = prefetch_model
        self._loader = loader
        self._prepare = prepare
        self._lock = RLock()
        self._runtime = None
        self._variant = None
        self._vae = None

    def normalize_loras(self, raw):
        if not raw:
            return () if raw is None or raw == [] else LoRACatalog({}).normalize(raw)
        return LoRACatalog.scan(resolve_lora_directory()).normalize(raw)

    def close(self):
        """Release all Qwen components before another image model loads."""
        with self._lock:
            self._vae = None
            self._unload()

    def _unload(self):
        runtime, self._runtime = self._runtime, None
        self._variant = None
        if runtime is not None:
            with log_stage("model_unload"):
                runtime.close()
                del runtime
                gc.collect()

    def _select(self, variant):
        # Check completeness even for a warm, resident pipeline. A removed
        # or truncated local file must be repaired before the next request.
        with self._prepare(variant) as pending:
            if self._variant != variant or pending is not None:
                self._unload()
                if pending is not None and getattr(pending, "reload_vae", False):
                    self._vae = None
                logger.info("Loading selected model_variant=%s", variant)
                with startup_diagnostics():
                    self._runtime = self._loader(model_variant=variant, vae=self._vae)
                self._variant = variant
                if resolve_vae_variant() == "texture-fix":
                    self._vae = self._runtime.pipe.vae
            if pending is not None:
                pending.wait("all")

    def run(self, norm, seed: int, loras=()):
        with self._lock:
            try:
                self._select(norm.model_variant)
            except BaseException:
                # A failed loader can leave offload hooks on shared modules.
                # Retry from complete files with fresh module instances.
                self._vae = None
                self._unload()
                raise
            return self._runtime.run(norm, seed=seed, loras=loras)
