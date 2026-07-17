"""LoRA discovery and request validation for the Krea 2 worker."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from safetensors.torch import load_file


logger = logging.getLogger(__name__)

MAX_LORAS_PER_REQUEST = 4
MIN_LORA_STRENGTH = 0.0
MAX_LORA_STRENGTH = 2.0

_COMPONENT_PREFIXES = ("diffusion_model.", "transformer.")
_PAIR_SUFFIXES = {
    ".lora_A.weight": ("down", "peft"),
    ".lora_B.weight": ("up", "peft"),
    ".lora_down.weight": ("down", "native"),
    ".lora_up.weight": ("up", "native"),
}
_ALPHA_SUFFIX = ".alpha"

_BLOCK_TARGET_MAPPINGS = {
    "attn.to_q": "attn.wq",
    "attn.to_k": "attn.wk",
    "attn.to_v": "attn.wv",
    "attn.to_gate": "attn.gate",
    "attn.to_out.0": "attn.wo",
    "attn.to_out": "attn.wo",
    "ff.gate": "mlp.gate",
    "ff.up": "mlp.up",
    "ff.down": "mlp.down",
}
_BASIC_TARGET_MAPPINGS = {
    "img_in": "first",
    "time_embed.linear_1": "tmlp.0",
    "time_embed.linear_2": "tmlp.2",
    "time_mod_proj": "tproj.1",
    "txt_in.linear_1": "txtmlp.1",
    "txt_in.linear_2": "txtmlp.3",
    "text_fusion.projector": "txtfusion.projector",
    "final_layer.linear": "last.linear",
}
_BLOCK_TARGET_PATTERN = re.compile(
    r"^(?P<family>transformer_blocks|text_fusion\.(?:layerwise_blocks|refiner_blocks))"
    r"\.(?P<index>\d+)\.(?P<target>.+)$"
)


class LoRAError(ValueError):
    """A validation error whose message is safe to return to a client."""


@dataclass(frozen=True)
class LoRASelection:
    name: str
    strength: float
    path: Path

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "strength": self.strength}


@dataclass(frozen=True)
class LoRALayerWeights:
    target: str
    down: torch.Tensor
    up: torch.Tensor
    scale: float


@dataclass(frozen=True)
class PreparedLoRA:
    name: str
    strength: float
    layers: tuple[LoRALayerWeights, ...]


@dataclass(frozen=True)
class LoRACatalog:
    root: Path
    _paths: Mapping[str, Path]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_paths", MappingProxyType(dict(self._paths)))

    @classmethod
    def scan(cls, directory: Path | str) -> "LoRACatalog":
        root = Path(directory)
        if not root.exists():
            logger.warning("LoRA directory does not exist: %s", root)
            return cls(root=root, _paths={})
        if not root.is_dir():
            raise RuntimeError(f"LoRA path is not a directory: {root}")

        paths: dict[str, Path] = {}
        for path in sorted(root.glob("*.safetensors")):
            if path.is_symlink() or not path.is_file():
                continue
            name = path.name.removesuffix(".safetensors")
            if not _is_safe_name(name):
                raise RuntimeError(f"Invalid LoRA ID: {name!r}")
            if name in paths:
                raise RuntimeError(f"Duplicate LoRA ID: {name}")
            paths[name] = path

        logger.info("Discovered %d LoRA(s) in %s", len(paths), root)
        return cls(root=root, _paths=paths)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._paths))

    def resolve(self, name: str) -> Path:
        try:
            return self._paths[name]
        except KeyError:
            raise LoRAError(f"Unknown LoRA: {name}") from None


def normalize_lora_requests(
    raw: object,
    catalog: LoRACatalog,
) -> tuple[LoRASelection, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise LoRAError("LoRAs must be a list")
    if len(raw) > MAX_LORAS_PER_REQUEST:
        raise LoRAError(f"At most {MAX_LORAS_PER_REQUEST} LoRAs may be requested")

    seen: set[str] = set()
    selections: list[LoRASelection] = []
    for item in raw:
        if not isinstance(item, dict):
            raise LoRAError("Each LoRA must be an object")
        if "name" not in item or not set(item).issubset({"name", "strength"}):
            raise LoRAError("Each LoRA object must contain only name and strength")

        name = item["name"]
        if not isinstance(name, str) or not _is_safe_name(name):
            raise LoRAError("Invalid LoRA name")
        if name.endswith(".safetensors"):
            raise LoRAError("LoRA names must not include the .safetensors suffix")
        if name in seen:
            raise LoRAError(f"Duplicate LoRA: {name}")
        seen.add(name)

        raw_strength = item.get("strength", 1.0)
        if isinstance(raw_strength, bool) or not isinstance(raw_strength, (int, float)):
            raise LoRAError("LoRA strength must be a number")
        try:
            strength = float(raw_strength)
        except (OverflowError, ValueError):
            raise LoRAError("LoRA strength must be finite") from None
        if not math.isfinite(strength):
            raise LoRAError("LoRA strength must be finite")
        if not MIN_LORA_STRENGTH <= strength <= MAX_LORA_STRENGTH:
            raise LoRAError(
                f"LoRA strength must be between {MIN_LORA_STRENGTH} "
                f"and {MAX_LORA_STRENGTH}"
            )

        path = catalog.resolve(name)
        if strength != 0.0:
            selections.append(LoRASelection(name=name, strength=strength, path=path))

    return tuple(selections)


class LoRALoader:
    """Load and validate standard low-rank LoRA weights for a Krea 2 model."""

    def __init__(self, model: nn.Module):
        self._linear_targets = {
            name: module
            for name, module in model.named_modules()
            if name and isinstance(module, nn.Linear)
        }

    def load(self, selections: Sequence[LoRASelection]) -> tuple[PreparedLoRA, ...]:
        return tuple(self._load_one(selection) for selection in selections)

    def _load_one(self, selection: LoRASelection) -> PreparedLoRA:
        try:
            tensors = load_file(str(selection.path), device="cpu")
        except Exception:
            raise LoRAError(f"Could not load LoRA: {selection.name}") from None

        grouped: dict[str, dict[str, torch.Tensor]] = {}
        pair_conventions: dict[str, str] = {}
        for key in sorted(tensors):
            parsed = self._parse_tensor_key(key)
            if parsed is None:
                self._invalid(selection, "unsupported tensor key")
            base, part, convention = parsed
            parts = grouped.setdefault(base, {})
            if part in parts:
                self._invalid(selection, "duplicate weight tensor")
            if convention is not None:
                previous_convention = pair_conventions.setdefault(base, convention)
                if previous_convention != convention:
                    self._invalid(selection, "mixed weight conventions")
            parts[part] = tensors[key]

        layers: list[LoRALayerWeights] = []
        resolved_targets: set[str] = set()
        for base, parts in grouped.items():
            if "down" not in parts and "up" not in parts and "alpha" in parts:
                self._invalid(selection, "unmatched alpha")
            if "down" not in parts or "up" not in parts:
                self._invalid(selection, "incomplete weight pair")

            target = self._resolve_target(base)
            if target is None:
                self._invalid(selection, "unknown target")
            if target in resolved_targets:
                self._invalid(selection, "duplicate target")
            resolved_targets.add(target)

            down = parts["down"]
            up = parts["up"]
            module = self._linear_targets[target]
            rank = self._validate_dimensions(selection, module, down, up)
            alpha = self._validate_alpha(selection, parts.get("alpha"), rank)
            layers.append(
                LoRALayerWeights(
                    target=target,
                    down=down,
                    up=up,
                    scale=alpha / rank,
                )
            )

        if not layers:
            self._invalid(selection, "contains no LoRA layers")

        layers.sort(key=lambda layer: layer.target)
        return PreparedLoRA(
            name=selection.name,
            strength=selection.strength,
            layers=tuple(layers),
        )

    @staticmethod
    def _parse_tensor_key(key: str) -> tuple[str, str, str | None] | None:
        for suffix, (part, convention) in _PAIR_SUFFIXES.items():
            if key.endswith(suffix):
                return key[: -len(suffix)], part, convention
        if key.endswith(_ALPHA_SUFFIX):
            return key[: -len(_ALPHA_SUFFIX)], "alpha", None
        return None

    def _resolve_target(self, source: str) -> str | None:
        target = source
        for prefix in _COMPONENT_PREFIXES:
            if target.startswith(prefix):
                target = target[len(prefix) :]
                break

        if target in self._linear_targets:
            return target

        mapped = _BASIC_TARGET_MAPPINGS.get(target)
        if mapped is None:
            match = _BLOCK_TARGET_PATTERN.fullmatch(target)
            if match is None:
                return None
            mapped_target = _BLOCK_TARGET_MAPPINGS.get(match.group("target"))
            if mapped_target is None:
                return None
            family = match.group("family")
            if family == "transformer_blocks":
                root = "blocks"
            else:
                root = f"txtfusion.{family.removeprefix('text_fusion.')}"
            mapped = f"{root}.{match.group('index')}.{mapped_target}"

        return mapped if mapped in self._linear_targets else None

    @staticmethod
    def _validate_dimensions(
        selection: LoRASelection,
        module: nn.Linear,
        down: torch.Tensor,
        up: torch.Tensor,
    ) -> int:
        if not down.is_floating_point() or not up.is_floating_point():
            LoRALoader._invalid(selection, "non-floating-point weights")
        if down.ndim != 2 or up.ndim != 2:
            LoRALoader._invalid(selection, "non-matrix weights")
        rank = down.shape[0]
        if rank <= 0:
            LoRALoader._invalid(selection, "invalid rank")
        if down.shape[1] != module.in_features:
            LoRALoader._invalid(selection, "input dimension mismatch")
        if up.shape[0] != module.out_features:
            LoRALoader._invalid(selection, "output dimension mismatch")
        if up.shape[1] != rank:
            LoRALoader._invalid(selection, "rank mismatch")
        return rank

    @staticmethod
    def _validate_alpha(
        selection: LoRASelection,
        alpha_tensor: torch.Tensor | None,
        rank: int,
    ) -> float:
        if alpha_tensor is None:
            return float(rank)
        if alpha_tensor.dtype == torch.bool:
            LoRALoader._invalid(selection, "boolean alpha")
        if alpha_tensor.ndim != 0:
            LoRALoader._invalid(selection, "non-scalar alpha")
        try:
            alpha = float(alpha_tensor.item())
        except (OverflowError, TypeError, ValueError):
            LoRALoader._invalid(selection, "invalid alpha")
        if not math.isfinite(alpha):
            LoRALoader._invalid(selection, "non-finite alpha")
        return alpha

    @staticmethod
    def _invalid(selection: LoRASelection, reason: str) -> None:
        raise LoRAError(f"Invalid LoRA {selection.name}: {reason}")


def _is_safe_name(name: str) -> bool:
    return bool(name) and name not in {".", ".."} and "/" not in name and "\\" not in name
