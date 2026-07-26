"""Exception-safe runtime application of Krea 2 LoRA weights."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import (
    LoRACatalog,
    LoRAError,
    LoRALayerWeights,
    LoRALoader,
    LoRASelection,
    PreparedLoRA,
    WeightDiffLayerWeights,
    normalize_lora_requests,
)

logger = logging.getLogger(__name__)


_ACTIVE_ATTR = "_krea2_active_lora_layers"
_PATCHED_ATTR = "_krea2_lora_compute_dtype"
_REGISTRY_ATTR = "_krea2_lora_runtime_registry"
_MISSING_TARGET = object()

_FP8_DTYPES = frozenset(
    dtype
    for name in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
    )
    if (dtype := getattr(torch, name, None)) is not None
)


def is_fp8_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype in _FP8_DTYPES


def _detach_exception_references(error: BaseException) -> BaseException:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        cause = current.__cause__
        context = current.__context__
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None
        if cause is not None:
            pending.append(cause)
        if context is not None:
            pending.append(context)
    return error


@dataclass(frozen=True)
class ActiveLoRALayer:
    multiplier: float
    down: torch.Tensor | None = None
    up: torch.Tensor | None = None
    delta: torch.Tensor | None = None


def _patch_linear(module: nn.Linear, compute_dtype: torch.dtype) -> None:
    if hasattr(module, _PATCHED_ATTR):
        setattr(module, _PATCHED_ATTR, compute_dtype)
        return

    setattr(module, _ACTIVE_ATTR, ())
    setattr(module, _PATCHED_ATTR, compute_dtype)

    def forward(x: torch.Tensor) -> torch.Tensor:
        dtype = getattr(module, _PATCHED_ATTR)
        if is_fp8_tensor(module.weight):
            weight = module.weight.to(dtype=dtype)
            bias = None if module.bias is None else module.bias.to(dtype=dtype)
            output = F.linear(x.to(dtype=dtype), weight, bias)
        else:
            output = F.linear(x, module.weight, module.bias)

        active_layers: tuple[ActiveLoRALayer, ...] = getattr(module, _ACTIVE_ATTR)
        if active_layers:
            lora_input = x.to(dtype=dtype)
            for layer in active_layers:
                if layer.delta is not None:
                    delta = F.linear(lora_input, layer.delta)
                else:
                    assert layer.down is not None and layer.up is not None
                    low_rank = F.linear(lora_input, layer.down)
                    delta = F.linear(low_rank, layer.up)
                output = output + delta.to(dtype=output.dtype) * layer.multiplier
        return output

    module.forward = forward  # type: ignore[method-assign]


class RuntimeLinearRegistry:
    """Patched Linear targets and their serialized activation state."""

    def __init__(
        self,
        modules: dict[str, nn.Linear],
        compute_dtype: torch.dtype,
    ) -> None:
        self._modules = modules
        self.compute_dtype = compute_dtype
        self._lock = threading.Lock()

    @classmethod
    def patch(
        cls,
        model: nn.Module,
        compute_dtype: torch.dtype,
    ) -> "RuntimeLinearRegistry":
        existing = getattr(model, _REGISTRY_ATTR, None)
        if existing is not None:
            if not isinstance(existing, cls):
                raise RuntimeError("Model has incompatible LoRA runtime state")
            if existing.compute_dtype != compute_dtype:
                raise ValueError(
                    "LoRA runtime compute dtype does not match the existing registry"
                )
            return existing

        modules = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        }
        for module in modules.values():
            _patch_linear(module, compute_dtype)
        registry = cls(modules, compute_dtype)
        setattr(model, _REGISTRY_ATTR, registry)
        return registry

    def activate(self, prepared: Sequence[PreparedLoRA]) -> "LoRAActivation":
        return LoRAActivation(self, prepared)


class LoRAActivation:
    """Install prepared LoRA weights for one serialized inference scope."""

    def __init__(
        self,
        registry: RuntimeLinearRegistry,
        prepared: Sequence[PreparedLoRA],
    ) -> None:
        self._registry = registry
        self._prepared = tuple(prepared)
        self._touched: list[nn.Linear] = []
        self._lock_acquired = False

    def __enter__(self) -> "LoRAActivation":
        if self._lock_acquired:
            self._clear()
            raise RuntimeError("LoRA activation is already active")

        self._registry._lock.acquire()
        self._lock_acquired = True
        grouped, error = self._prepare_active_layers()
        if error is not None:
            self._clear()
            raise error from None

        try:
            for module, layers in grouped.items():
                setattr(module, _ACTIVE_ATTR, tuple(layers))
                self._touched.append(module)
            return self
        except BaseException:
            self._clear()
            raise
        finally:
            grouped.clear()

    def _prepare_active_layers(
        self,
    ) -> tuple[
        dict[nn.Linear, list[ActiveLoRALayer]],
        None,
    ] | tuple[None, BaseException]:
        grouped: dict[nn.Linear, list[ActiveLoRALayer]] = {}
        module = None
        down = None
        up = None
        delta = None
        active = None
        try:
            for adapter in self._prepared:
                for layer in adapter.layers:
                    module = self._registry._modules.get(
                        layer.target,
                        _MISSING_TARGET,
                    )
                    if module is _MISSING_TARGET:
                        raise LoRAError(f"Unknown LoRA target: {layer.target}") from None
                    if isinstance(layer, WeightDiffLayerWeights):
                        delta = layer.delta.to(
                            device=module.weight.device,
                            dtype=self._registry.compute_dtype,
                        )
                        active = ActiveLoRALayer(
                            multiplier=adapter.strength,
                            delta=delta,
                        )
                    elif isinstance(layer, LoRALayerWeights):
                        down = layer.down.to(
                            device=module.weight.device,
                            dtype=self._registry.compute_dtype,
                        )
                        up = layer.up.to(
                            device=module.weight.device,
                            dtype=self._registry.compute_dtype,
                        )
                        active = ActiveLoRALayer(
                            multiplier=adapter.strength * layer.scale,
                            down=down,
                            up=up,
                        )
                    else:
                        raise LoRAError(
                            f"Unsupported LoRA layer type for {adapter.name}"
                        ) from None
                    grouped.setdefault(module, []).append(active)
                    module = None
                    down = None
                    up = None
                    delta = None
                    active = None
            return grouped, None
        except BaseException as error:
            grouped.clear()
            module = None
            down = None
            up = None
            delta = None
            active = None
            return None, _detach_exception_references(error)

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self._clear()
        return False

    def _clear(self) -> None:
        try:
            for module in self._touched:
                setattr(module, _ACTIVE_ATTR, ())
            self._touched.clear()
            self._prepared = ()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            if self._lock_acquired:
                self._lock_acquired = False
                self._registry._lock.release()


class LoRAManager:
    """Normalize, load, and activate request-scoped LoRA adapters."""

    def __init__(
        self,
        model: nn.Module,
        catalog: LoRACatalog,
        compute_dtype: torch.dtype,
    ) -> None:
        self.catalog = catalog
        self._loader = LoRALoader(model)
        self._registry = RuntimeLinearRegistry.patch(model, compute_dtype)

    def normalize(self, raw: object) -> tuple[LoRASelection, ...]:
        return normalize_lora_requests(raw, self.catalog)

    def load_prepared(
        self, selections: Sequence[LoRASelection]
    ) -> tuple[PreparedLoRA, ...]:
        prepared = self._loader.load(selections)
        for adapter in prepared:
            logger.info(
                "Loaded LoRA %s type=%s strength=%.3f layers=%d",
                adapter.name,
                adapter.type,
                adapter.strength,
                len(adapter.layers),
            )
        return prepared

    def activation(self, selections: Sequence[LoRASelection]) -> LoRAActivation:
        prepared = self.load_prepared(selections)
        return self._registry.activate(prepared)
