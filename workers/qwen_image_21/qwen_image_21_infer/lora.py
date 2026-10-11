"""Local, per-request Qwen LoRAs using the Krea 2 request contract."""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import MappingProxyType

from .runtime_config import LORA_WEIGHT_NAME

logger = logging.getLogger(__name__)
TURBO_ADAPTER_NAME = "viggle_turbo"
MAX_LORAS_PER_REQUEST = 4


class LoRAError(ValueError):
    """Safe client-facing error; invalid adapters do not require a worker restart."""


@dataclass(frozen=True)
class LoRASelection:
    name: str
    strength: float
    path: Path
    type: str = "lora"

    def as_dict(self) -> dict:
        return {"name": self.name, "strength": self.strength, "type": self.type}


def _safe_name(name) -> bool:
    return (
        isinstance(name, str)
        and bool(name)
        and name not in {".", ".."}
        and not any(char in name for char in ("/", "\\", "\x00"))
    )


class LoRACatalog:
    def __init__(self, paths):
        self._paths = MappingProxyType(dict(paths))

    @classmethod
    def scan(cls, directory: Path | str):
        root = Path(directory)
        if not root.exists():
            logger.warning("LoRA directory does not exist: %s", root)
            return cls({})
        if not root.is_dir():
            raise RuntimeError(f"LoRA path is not a directory: {root}")
        paths = {}
        for path in sorted(root.glob("*.safetensors")):
            if path.name == LORA_WEIGHT_NAME or path.is_symlink() or not path.is_file():
                continue
            if not _safe_name(path.stem):
                raise RuntimeError(f"Invalid LoRA ID: {path.stem!r}")
            paths[path.stem] = path
        logger.info("Discovered %d additional LoRA(s) in %s", len(paths), root)
        return cls(paths)

    def normalize(self, raw) -> tuple[LoRASelection, ...]:
        if raw is None:
            return ()
        if not isinstance(raw, list):
            raise LoRAError("LoRAs must be a list")
        if len(raw) > MAX_LORAS_PER_REQUEST:
            raise LoRAError(f"At most {MAX_LORAS_PER_REQUEST} LoRAs may be requested")
        selections, seen = [], set()
        for item in raw:
            if not isinstance(item, dict):
                raise LoRAError("Each LoRA must be an object")
            if set(item) - {"name", "strength", "type"}:
                raise LoRAError(
                    "Each LoRA object must contain only name, strength, and type"
                )
            name = item.get("name")
            if not _safe_name(name):
                raise LoRAError("Invalid LoRA name")
            if name.endswith(".safetensors"):
                raise LoRAError("LoRA names must not include the .safetensors suffix")
            if name in seen:
                raise LoRAError(f"Duplicate LoRA: {name}")
            seen.add(name)
            if item.get("type", "lora") != "lora":
                raise LoRAError(
                    "Qwen-Image-2.1 supports only LoRA type 'lora'; weight_diff is unsupported"
                )
            strength = item.get("strength", 1.0)
            if isinstance(strength, bool) or not isinstance(strength, (int, float)):
                raise LoRAError("LoRA strength must be a number")
            try:
                strength = float(strength)
            except OverflowError:
                raise LoRAError("LoRA strength must be finite") from None
            if not math.isfinite(strength):
                raise LoRAError("LoRA strength must be finite")
            if not 0 <= strength <= 2:
                raise LoRAError("LoRA strength must be between 0 and 2")
            if name not in self._paths:
                raise LoRAError(f"Unknown LoRA: {name}")
            if strength:
                selections.append(LoRASelection(name, strength, self._paths[name]))
        return tuple(selections)


class LoRAManager:
    """Serialize inference and remove additional adapters after every job."""

    def __init__(self, pipe, catalog: LoRACatalog, *, offload: str | None = None):
        self.pipe = pipe
        self.catalog = catalog
        self._lock = RLock()
        self._offload = offload

    def normalize(self, raw):
        return self.catalog.normalize(raw)

    @contextmanager
    def activate(self, selections):
        with self._lock:
            names = []
            try:
                for index, selection in enumerate(selections):
                    # Internal IDs are independent of filenames and cannot collide with Viggle.
                    adapter_name = f"request_lora_{index}"
                    names.append(adapter_name)
                    self._load(selection, adapter_name)
                self.pipe.set_adapters(
                    [TURBO_ADAPTER_NAME, *names],
                    adapter_weights=[1.0, *(item.strength for item in selections)],
                )
                yield
            finally:
                self.pipe.set_adapters([TURBO_ADAPTER_NAME], adapter_weights=[1.0])
                if names:
                    # A failed PEFT injection can drop its config while leaving
                    # layer weights. Diffusers deletion also handles absent IDs.
                    self.pipe.delete_adapters(names)

    def _load(self, selection: LoRASelection, adapter_name: str):
        path = selection.path
        if path.is_symlink() or not path.is_file():
            raise LoRAError(f"LoRA file is unavailable: {selection.name}")
        try:
            state, metadata = self.pipe.lora_state_dict(
                str(path.parent),
                weight_name=path.name,
                local_files_only=True,
                use_safetensors=True,
                return_lora_metadata=True,
            )
        except (RuntimeError, MemoryError):
            raise
        except Exception:
            logger.exception("Could not read LoRA %s", selection.name)
            raise LoRAError(f"Could not load LoRA: {selection.name}") from None
        state, metadata = _split_fused_mlp_lora(state, metadata, selection.name)
        self._validate(state, selection.name)
        attached = False
        try:
            # Use the converted state once and retain safetensors metadata (alpha/rank).
            # Passing _pipeline lets Diffusers remove and reinstall CPU offload hooks.
            self.pipe.load_lora_into_transformer(
                state,
                transformer=self.pipe.transformer,
                adapter_name=adapter_name,
                metadata=metadata,
                _pipeline=self.pipe,
                low_cpu_mem_usage=True,
            )
            attached = True
        except (ValueError, KeyError, TypeError):
            logger.exception("Could not attach LoRA %s", selection.name)
            raise LoRAError(f"Incompatible LoRA: {selection.name}") from None
        finally:
            # Diffusers reinstalls hooks after a successful load, but its failure
            # path can leave them removed. Keep the worker usable after bad files.
            if not attached:
                if self._offload == "cpu":
                    self.pipe.enable_model_cpu_offload()
                elif self._offload == "sequential":
                    self.pipe.enable_sequential_cpu_offload()

    def _validate(self, state, name):
        def invalid(reason):
            raise LoRAError(f"Invalid LoRA {name}: {reason}")

        if not state:
            invalid("contains no LoRA layers")
        pairs = {}
        for key, tensor in state.items():
            suffix = next(
                (
                    part
                    for part in (".lora_A.weight", ".lora_B.weight")
                    if key.endswith(part)
                ),
                None,
            )
            if not key.startswith("transformer.") or suffix is None:
                invalid("only transformer LoRA A/B weights are supported")
            if tensor.ndim != 2 or not tensor.is_floating_point():
                invalid("LoRA weights must be floating-point matrices")
            target = key[len("transformer.") : -len(suffix)]
            pairs.setdefault(target, {})[suffix] = tensor
        modules = dict(self.pipe.transformer.named_modules())
        for target, weights in pairs.items():
            module = modules.get(target)
            if (
                module is None
                or not hasattr(module, "in_features")
                or not hasattr(module, "out_features")
            ):
                invalid("target layer is incompatible with Qwen-Image-2.1")
            down, up = weights.get(".lora_A.weight"), weights.get(".lora_B.weight")
            if down is None or up is None:
                invalid("missing LoRA A/B pair")
            rank, inputs = down.shape
            outputs, up_rank = up.shape
            if (
                rank < 1
                or up_rank != rank
                or inputs != module.in_features
                or outputs != module.out_features
            ):
                invalid("weight shape mismatch")


def _split_fused_mlp_lora(state, metadata, name):
    """Map ComfyUI's Qwen 2.1 [gate; up] LoRA to Diffusers' two linears."""
    suffixes = (".lora_A.weight", ".lora_B.weight")
    targets = {
        key[: -len(suffix)]
        for key in state
        for suffix in suffixes
        if key.startswith("transformer.") and key.endswith(f".img_mlp.gate_up{suffix}")
    }
    if not targets:
        return state, metadata
    converted = dict(state)
    for target in sorted(targets):
        down, up = (state.get(target + suffix) for suffix in suffixes)
        if (
            down is None or up is None or down.ndim != 2 or up.ndim != 2
            or not down.is_floating_point() or not up.is_floating_point()
            or down.shape[0] < 1 or up.shape[1] != down.shape[0]
            or up.shape[0] < 2 or up.shape[0] % 2
        ):
            raise LoRAError(f"Invalid LoRA {name}: malformed fused MLP A/B weights")
        branches = [target.removesuffix("gate_up") + leaf for leaf in ("gate_layer", "proj")]
        if any(branch + suffix in state for branch in branches for suffix in suffixes):
            raise LoRAError(f"Invalid LoRA {name}: conflicting fused MLP weights")
        # B has output rows [gate; up]; A's input projection is shared.
        for branch, rows in zip(branches, up.chunk(2, dim=0)):
            converted[branch + suffixes[0]] = down
            converted[branch + suffixes[1]] = rows.contiguous()
        for suffix in suffixes:
            del converted[target + suffix]
    if metadata:
        metadata = dict(metadata)
        # PEFT metadata can name the fused target explicitly. Retain scaling,
        # but make target selection and rank/alpha patterns follow the split.
        metadata["transformer.target_modules"] = [
            key.removeprefix("transformer.").removesuffix(suffixes[0])
            for key in converted if key.endswith(suffixes[0])
        ]
        for field in ("transformer.rank_pattern", "transformer.alpha_pattern"):
            if metadata.get(field):
                patterns = {}
                for key, value in metadata[field].items():
                    if key in {"gate_up", "img_mlp.gate_up"} or key.endswith(".img_mlp.gate_up"):
                        for leaf in ("gate_layer", "proj"):
                            patterns[key.removesuffix("gate_up") + leaf] = value
                    else:
                        patterns[key] = value
                metadata[field] = patterns
    logger.info("Converted %d fused MLP LoRA target(s) for %s", len(targets), name)
    return converted, metadata
