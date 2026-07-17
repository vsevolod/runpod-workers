"""LoRA discovery and request validation for the Krea 2 worker."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


logger = logging.getLogger(__name__)

MAX_LORAS_PER_REQUEST = 4
MIN_LORA_STRENGTH = 0.0
MAX_LORA_STRENGTH = 2.0


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


def _is_safe_name(name: str) -> bool:
    return bool(name) and name not in {".", ".."} and "/" not in name and "\\" not in name
