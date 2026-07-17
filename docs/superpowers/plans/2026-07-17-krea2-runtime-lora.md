# Krea 2 Runtime LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Добавить в thin Krea 2 RunPod worker выбор до четырёх заранее размещённых text-to-image LoRA с отдельным strength без изменения базовых FP8-весов.

**Architecture:** Worker при старте сканирует `$LORA_DIR/*.safetensors` в allowlist, а на job читает только выбранные файлы. Runtime-обвязка каждого `nn.Linear` вычисляет базовый FP8/BF16 output и суммирует BF16 `B(A(x)) * alpha/rank * strength`; контекст активации существует только во время DiT denoise и снимается до VAE decode.

**Tech Stack:** Python 3.12, PyTorch, safetensors, `unittest`, RunPod serverless validator, Docker.

---

## File map

- Create `workers/krea2/krea2_infer/lora.py`: каталог, API-нормализация, разбор safetensors, Krea 2 key mapping и CPU-представление адаптеров.
- Create `workers/krea2/krea2_infer/lora_runtime.py`: LoRA-compatible Linear forward, process-local lock и lifecycle GPU activation.
- Create `workers/krea2/tests/__init__.py`: package marker для тестов.
- Create `workers/krea2/tests/test_lora.py`: каталог, входной контракт, key mapping и shape validation.
- Create `workers/krea2/tests/test_lora_runtime.py`: математика нескольких runtime-адаптеров и cleanup.
- Create `workers/krea2/tests/test_sampling_lora.py`: граница activation/deactivation вокруг denoise.
- Modify `workers/krea2/krea2_infer/pipeline.py:16-24,58-102,105-146,191-304`: заменить FP8-only monkey patch на LoRA-compatible runtime и собрать `LoRAManager`.
- Modify `workers/krea2/krea2_infer/sampling.py:86-189`: принимать activation context и завершать его до VAE decode.
- Modify `workers/krea2/schemas.py:3-52`: добавить top-level list `loras`.
- Modify `workers/krea2/handler.py:20-21,86-157`: нормализовать LoRA, передать их pipeline, обработать безопасные ошибки и вернуть applied list.
- Modify `workers/krea2/Dockerfile:56-59`: задать стандартный `LORA_DIR`.
- Modify `workers/krea2/README.md:13-186`: описать volume layout, env, API, ограничения и lifecycle.

## Test environment

Локальное окружение не содержит PyTorch. Перед первым red/green циклом создать одноразовый CPU venv вне репозитория:

```bash
python -m venv /tmp/runpod-krea2-lora-venv
/tmp/runpod-krea2-lora-venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
/tmp/runpod-krea2-lora-venv/bin/pip install safetensors pillow einops
```

Все unit-команды ниже запускаются из корня репозитория с:

```bash
PYTHONPATH=workers/krea2 /tmp/runpod-krea2-lora-venv/bin/python -m unittest discover -s workers/krea2/tests -p 'test_*.py' -v
```

Не добавлять venv, реальные LoRA или их имена в git.

Перед Task 1 вызвать `superpowers:using-git-worktrees` и создать отдельный
feature worktree от commit с этой plan/spec. Все последующие task-коммиты делать
в этом worktree, не в `master`.

### Task 1: Startup catalog and request contract

**Files:**
- Create: `workers/krea2/krea2_infer/lora.py`
- Create: `workers/krea2/tests/__init__.py`
- Create: `workers/krea2/tests/test_lora.py`

- [ ] **Step 1: Write failing catalog and normalization tests**

Create an empty `workers/krea2/tests/__init__.py`, then create `workers/krea2/tests/test_lora.py`:

```python
import tempfile
import unittest
from pathlib import Path

from krea2_infer.lora import (
    LoRACatalog,
    LoRAError,
    normalize_lora_requests,
)


class LoRACatalogTests(unittest.TestCase):
    def test_scan_uses_only_top_level_safetensors_stems(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "lora_a.safetensors").touch()
            (root / "ignore.txt").touch()
            (root / "nested").mkdir()
            (root / "nested" / "lora_b.safetensors").touch()

            catalog = LoRACatalog.scan(root)

            self.assertEqual(catalog.names, ("lora_a",))
            self.assertEqual(catalog.resolve("lora_a"), root / "lora_a.safetensors")

    def test_missing_directory_is_an_empty_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = LoRACatalog.scan(Path(tmp) / "missing")
            self.assertEqual(catalog.names, ())


class NormalizeLoRARequestsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        for name in ("lora_a", "lora_b", "lora_c", "lora_d"):
            (root / f"{name}.safetensors").touch()
        self.catalog = LoRACatalog.scan(root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_defaults_strength_and_skips_zero_strength(self):
        selections = normalize_lora_requests(
            [
                {"name": "lora_a"},
                {"name": "lora_b", "strength": 0.0},
                {"name": "lora_c", "strength": 0.75},
            ],
            self.catalog,
        )
        self.assertEqual(
            [selection.as_dict() for selection in selections],
            [
                {"name": "lora_a", "strength": 1.0},
                {"name": "lora_c", "strength": 0.75},
            ],
        )

    def test_rejects_more_than_four_before_skipping_zero(self):
        request = [{"name": "lora_a", "strength": 0.0}] * 5
        with self.assertRaisesRegex(LoRAError, "at most 4"):
            normalize_lora_requests(request, self.catalog)

    def test_rejects_duplicate_names(self):
        with self.assertRaisesRegex(LoRAError, "duplicate"):
            normalize_lora_requests(
                [{"name": "lora_a"}, {"name": "lora_a", "strength": 0.5}],
                self.catalog,
            )

    def test_rejects_unknown_name_without_exposing_catalog_path(self):
        with self.assertRaises(LoRAError) as error:
            normalize_lora_requests([{"name": "unknown"}], self.catalog)
        self.assertIn("unknown", str(error.exception))
        self.assertNotIn(self.tmp.name, str(error.exception))

    def test_rejects_path_components_extra_fields_and_bad_strengths(self):
        bad_requests = (
            [{"name": "../lora_a"}],
            [{"name": "lora_a.safetensors"}],
            [{"name": "lora_a", "unknown": 1}],
            [{"name": "lora_a", "strength": True}],
            [{"name": "lora_a", "strength": float("nan")}],
            [{"name": "lora_a", "strength": -0.1}],
            [{"name": "lora_a", "strength": 2.1}],
        )
        for request in bad_requests:
            with self.subTest(request=request):
                with self.assertRaises(LoRAError):
                    normalize_lora_requests(request, self.catalog)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify the expected import failure**

Run:

```bash
PYTHONPATH=workers/krea2 /tmp/runpod-krea2-lora-venv/bin/python -m unittest workers.krea2.tests.test_lora -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'krea2_infer.lora'`.

- [ ] **Step 3: Implement catalog and request normalization**

Create `workers/krea2/krea2_infer/lora.py` with:

```python
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_LORAS_PER_REQUEST = 4
MIN_LORA_STRENGTH = 0.0
MAX_LORA_STRENGTH = 2.0


class LoRAError(ValueError):
    """Safe client-facing error for LoRA selection or file contents."""


@dataclass(frozen=True)
class LoRASelection:
    name: str
    strength: float
    path: Path

    def as_dict(self) -> dict[str, str | float]:
        return {"name": self.name, "strength": self.strength}


@dataclass(frozen=True)
class LoRACatalog:
    root: Path
    _paths: dict[str, Path]

    @classmethod
    def scan(cls, directory: str | Path) -> "LoRACatalog":
        root = Path(directory)
        if not root.exists():
            logger.warning("LoRA directory does not exist; catalog is empty")
            return cls(root=root, _paths={})
        if not root.is_dir():
            raise RuntimeError("LORA_DIR must point to a directory")

        paths: dict[str, Path] = {}
        for candidate in sorted(root.glob("*.safetensors")):
            if not candidate.is_file():
                continue
            name = candidate.stem
            if (
                not name
                or name in {".", ".."}
                or "/" in name
                or "\\" in name
            ):
                raise RuntimeError("LORA_DIR contains an invalid safetensors name")
            if name in paths:
                raise RuntimeError(f"Duplicate LoRA id: {name}")
            paths[name] = candidate

        logger.info("Discovered %d LoRA file(s)", len(paths))
        return cls(root=root, _paths=paths)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._paths))

    def resolve(self, name: str) -> Path:
        try:
            return self._paths[name]
        except KeyError as err:
            raise LoRAError(f"Unknown LoRA: {name}") from err


def normalize_lora_requests(
    raw: Any,
    catalog: LoRACatalog,
) -> tuple[LoRASelection, ...]:
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise LoRAError("loras must be a list")
    if len(raw) > MAX_LORAS_PER_REQUEST:
        raise LoRAError("loras accepts at most 4 items")

    seen: set[str] = set()
    selections: list[LoRASelection] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise LoRAError(f"loras[{index}] must be an object")
        if set(item) - {"name", "strength"}:
            raise LoRAError(f"loras[{index}] contains unsupported fields")

        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise LoRAError(f"loras[{index}].name must be a non-empty string")
        if name.endswith(".safetensors") or "/" in name or "\\" in name:
            raise LoRAError(f"loras[{index}].name must be a catalog id")
        if name in seen:
            raise LoRAError(f"duplicate LoRA: {name}")
        seen.add(name)

        strength = item.get("strength", 1.0)
        if isinstance(strength, bool) or not isinstance(strength, (int, float)):
            raise LoRAError(f"loras[{index}].strength must be a number")
        strength = float(strength)
        if not math.isfinite(strength) or not (
            MIN_LORA_STRENGTH <= strength <= MAX_LORA_STRENGTH
        ):
            raise LoRAError(f"loras[{index}].strength must be between 0.0 and 2.0")

        path = catalog.resolve(name)
        if strength > 0.0:
            selections.append(LoRASelection(name, strength, path))

    return tuple(selections)
```

- [ ] **Step 4: Run the catalog tests and verify they pass**

Run the Task 1 unittest command. Expected: all `LoRACatalogTests` and `NormalizeLoRARequestsTests` PASS.

- [ ] **Step 5: Commit the catalog slice**

```bash
git add workers/krea2/krea2_infer/lora.py workers/krea2/tests/__init__.py workers/krea2/tests/test_lora.py
git commit -m "feat(krea2): add LoRA catalog and request validation"
```

### Task 2: Safetensors parser and Krea 2 key resolver

**Files:**
- Modify: `workers/krea2/krea2_infer/lora.py`
- Modify: `workers/krea2/tests/test_lora.py`

- [ ] **Step 1: Add failing loader tests**

Extend the import section and insert these test types before the final
`if __name__ == "__main__"` block in `workers/krea2/tests/test_lora.py`:

```python
import torch
import torch.nn as nn
from safetensors.torch import save_file

from krea2_infer.lora import LoRALoader, LoRASelection


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(3, 2, bias=False)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TinyAttention()


class TinyKrea2(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock()])


class LoRALoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = TinyKrea2()
        self.loader = LoRALoader(self.model)

    def tearDown(self):
        self.tmp.cleanup()

    def selection(self, filename: str, strength: float = 1.0) -> LoRASelection:
        return LoRASelection(filename.removesuffix(".safetensors"), strength, self.root / filename)

    def test_loads_diffusers_ab_keys_and_alpha(self):
        path = self.root / "lora_a.safetensors"
        save_file(
            {
                "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(1, 3),
                "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.ones(2, 1),
                "transformer.transformer_blocks.0.attn.to_q.alpha": torch.tensor(2.0),
            },
            path,
        )

        prepared = self.loader.load((self.selection(path.name, 0.5),))

        self.assertEqual(prepared[0].name, "lora_a")
        self.assertEqual(prepared[0].strength, 0.5)
        self.assertEqual(prepared[0].layers[0].target, "blocks.0.attn.wq")
        self.assertEqual(prepared[0].layers[0].scale, 2.0)

    def test_loads_native_up_down_keys_with_rank_default_alpha(self):
        path = self.root / "lora_b.safetensors"
        save_file(
            {
                "blocks.0.attn.wq.lora_down.weight": torch.ones(2, 3),
                "blocks.0.attn.wq.lora_up.weight": torch.ones(2, 2),
            },
            path,
        )

        prepared = self.loader.load((self.selection(path.name),))

        self.assertEqual(prepared[0].layers[0].scale, 1.0)

    def test_rejects_unknown_keys_partial_pairs_and_bad_shapes(self):
        cases = (
            {
                "transformer.unknown.lora_A.weight": torch.ones(1, 3),
                "transformer.unknown.lora_B.weight": torch.ones(2, 1),
            },
            {"blocks.0.attn.wq.lora_down.weight": torch.ones(1, 3)},
            {
                "blocks.0.attn.wq.lora_down.weight": torch.ones(1, 4),
                "blocks.0.attn.wq.lora_up.weight": torch.ones(2, 1),
            },
        )
        for index, tensors in enumerate(cases):
            path = self.root / f"bad_{index}.safetensors"
            save_file(tensors, path)
            with self.subTest(index=index):
                with self.assertRaises(LoRAError):
                    self.loader.load((self.selection(path.name),))
```

- [ ] **Step 2: Run only loader tests and verify the missing-symbol failure**

```bash
PYTHONPATH=workers/krea2 /tmp/runpod-krea2-lora-venv/bin/python -m unittest workers.krea2.tests.test_lora.LoRALoaderTests -v
```

Expected: FAIL importing `LoRALoader` from `krea2_infer.lora`.

- [ ] **Step 3: Implement exact Krea 2 name mapping and safe tensor parsing**

Add `import re`, `import torch`, `import torch.nn as nn`,
`from safetensors.torch import load_file`, and `Sequence` from `typing` to
`lora.py`, then add:

```python
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


_BLOCK_TARGETS = {
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

_BASIC_TARGETS = {
    "img_in": "first",
    "time_embed.linear_1": "tmlp.0",
    "time_embed.linear_2": "tmlp.2",
    "time_mod_proj": "tproj.1",
    "txt_in.linear_1": "txtmlp.1",
    "txt_in.linear_2": "txtmlp.3",
    "text_fusion.projector": "txtfusion.projector",
    "final_layer.linear": "last.linear",
}

_PAIR_SUFFIXES = {
    ".lora_A.weight": "down",
    ".lora_B.weight": "up",
    ".lora_down.weight": "down",
    ".lora_up.weight": "up",
}


def _without_supported_prefix(name: str) -> str:
    for prefix in ("diffusion_model.", "transformer."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _diffusers_target(name: str) -> str | None:
    if name in _BASIC_TARGETS:
        return _BASIC_TARGETS[name]

    match = re.fullmatch(r"transformer_blocks\.(\d+)\.(.+)", name)
    if match and match.group(2) in _BLOCK_TARGETS:
        return f"blocks.{match.group(1)}.{_BLOCK_TARGETS[match.group(2)]}"

    match = re.fullmatch(
        r"text_fusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\.(.+)",
        name,
    )
    if match and match.group(3) in _BLOCK_TARGETS:
        return (
            f"txtfusion.{match.group(1)}.{match.group(2)}."
            f"{_BLOCK_TARGETS[match.group(3)]}"
        )
    return None


class LoRALoader:
    def __init__(self, model: nn.Module):
        self._linears = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        }

    def _resolve_target(self, raw_target: str) -> str:
        target = _without_supported_prefix(raw_target)
        if target in self._linears:
            return target
        mapped = _diffusers_target(target)
        if mapped is None or mapped not in self._linears:
            raise LoRAError(f"Unsupported Krea 2 LoRA target: {raw_target}")
        return mapped

    def load(
        self,
        selections: Sequence[LoRASelection],
    ) -> tuple[PreparedLoRA, ...]:
        return tuple(self._load_one(selection) for selection in selections)

    def _load_one(self, selection: LoRASelection) -> PreparedLoRA:
        try:
            tensors = load_file(str(selection.path), device="cpu")
        except Exception as err:
            raise LoRAError(f"Invalid LoRA file: {selection.name}") from err

        pairs: dict[str, dict[str, torch.Tensor]] = {}
        alphas: dict[str, torch.Tensor] = {}
        consumed: set[str] = set()
        for key, tensor in tensors.items():
            matched = False
            for suffix, part in _PAIR_SUFFIXES.items():
                if key.endswith(suffix):
                    base = key[: -len(suffix)]
                    pair = pairs.setdefault(base, {})
                    if part in pair:
                        raise LoRAError(f"Duplicate LoRA tensor in {selection.name}")
                    pair[part] = tensor
                    consumed.add(key)
                    matched = True
                    break
            if not matched and key.endswith(".alpha"):
                alphas[key[: -len(".alpha")]] = tensor
                consumed.add(key)

        if set(tensors) != consumed:
            raise LoRAError(f"Unsupported LoRA tensors in {selection.name}")
        if not pairs:
            raise LoRAError(f"No LoRA layers found in {selection.name}")

        layers: list[LoRALayerWeights] = []
        for raw_target, pair in sorted(pairs.items()):
            if set(pair) != {"down", "up"}:
                raise LoRAError(f"Incomplete LoRA pair in {selection.name}")
            target = self._resolve_target(raw_target)
            module = self._linears[target]
            down, up = pair["down"], pair["up"]
            if down.ndim != 2 or up.ndim != 2:
                raise LoRAError(f"LoRA tensors must be matrices in {selection.name}")
            rank = down.shape[0]
            if (
                rank <= 0
                or down.shape[1] != module.in_features
                or up.shape[0] != module.out_features
                or up.shape[1] != rank
            ):
                raise LoRAError(f"LoRA shape mismatch in {selection.name}")

            alpha_tensor = alphas.pop(raw_target, None)
            if alpha_tensor is not None and alpha_tensor.numel() != 1:
                raise LoRAError(f"LoRA alpha must be scalar in {selection.name}")
            alpha = float(alpha_tensor.item()) if alpha_tensor is not None else float(rank)
            if not math.isfinite(alpha):
                raise LoRAError(f"LoRA alpha must be finite in {selection.name}")
            layers.append(LoRALayerWeights(target, down, up, alpha / rank))

        if alphas:
            raise LoRAError(f"Unmatched LoRA alpha in {selection.name}")
        return PreparedLoRA(selection.name, selection.strength, tuple(layers))
```

Ensure `Sequence` remains imported because `LoRALoader.load` now uses it.

- [ ] **Step 4: Run all lora parser tests**

Run:

```bash
PYTHONPATH=workers/krea2 /tmp/runpod-krea2-lora-venv/bin/python -m unittest workers.krea2.tests.test_lora -v
```

Expected: all catalog, normalization, and loader tests PASS.

- [ ] **Step 5: Commit parser and resolver**

```bash
git add workers/krea2/krea2_infer/lora.py workers/krea2/tests/test_lora.py
git commit -m "feat(krea2): parse standard Krea 2 LoRA weights"
```

### Task 3: Runtime Linear deltas and exception-safe activation

**Files:**
- Create: `workers/krea2/krea2_infer/lora_runtime.py`
- Create: `workers/krea2/tests/test_lora_runtime.py`

- [ ] **Step 1: Write failing runtime math and cleanup tests**

Create `workers/krea2/tests/test_lora_runtime.py`:

```python
import unittest

import torch
import torch.nn as nn

from krea2_infer.lora import LoRALayerWeights, PreparedLoRA
from krea2_infer.lora_runtime import RuntimeLinearRegistry


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(2))


def prepared(name, strength, down, up, scale=1.0):
    return PreparedLoRA(
        name=name,
        strength=strength,
        layers=(LoRALayerWeights("linear", down, up, scale),),
    )


class RuntimeLinearRegistryTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyModel()
        self.registry = RuntimeLinearRegistry.patch(self.model, torch.float32)
        self.x = torch.tensor([[2.0, 3.0]])

    def test_sums_multiple_runtime_deltas_without_mutating_base_weight(self):
        original_weight = self.model.linear.weight.detach().clone()
        first = prepared(
            "first",
            0.5,
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[2.0], [0.0]]),
            scale=2.0,
        )
        second = prepared(
            "second",
            1.0,
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[0.0], [1.0]]),
        )

        with self.registry.activate((first, second)):
            actual = self.model.linear(self.x)

        expected = torch.tensor([[6.0, 6.0]])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(self.model.linear.weight, original_weight)
        torch.testing.assert_close(self.model.linear(self.x), self.x)

    def test_cleans_runtime_state_and_releases_lock_after_exception(self):
        adapter = prepared(
            "first",
            1.0,
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0], [0.0]]),
        )

        with self.assertRaisesRegex(RuntimeError, "boom"):
            with self.registry.activate((adapter,)):
                raise RuntimeError("boom")

        torch.testing.assert_close(self.model.linear(self.x), self.x)
        with self.registry.activate((adapter,)):
            self.assertFalse(torch.equal(self.model.linear(self.x), self.x))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the runtime test and verify import failure**

```bash
PYTHONPATH=workers/krea2 /tmp/runpod-krea2-lora-venv/bin/python -m unittest workers.krea2.tests.test_lora_runtime -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'krea2_infer.lora_runtime'`.

- [ ] **Step 3: Implement LoRA-compatible forward and activation context**

Create `workers/krea2/krea2_infer/lora_runtime.py`:

```python
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import LoRALoader, LoRACatalog, LoRASelection, PreparedLoRA, normalize_lora_requests

_ACTIVE_ATTR = "_krea2_active_loras"
_PATCHED_ATTR = "_krea2_lora_patched"
_FP8_DTYPES = {
    dtype
    for name in ("float8_e4m3fn", "float8_e5m2")
    if (dtype := getattr(torch, name, None)) is not None
}


@dataclass(frozen=True)
class ActiveLoRALayer:
    down: torch.Tensor
    up: torch.Tensor
    multiplier: float


def is_fp8_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype in _FP8_DTYPES


def _patch_linear(module: nn.Linear, compute_dtype: torch.dtype) -> bool:
    if getattr(module, _PATCHED_ATTR, False):
        return False
    setattr(module, _ACTIVE_ATTR, ())
    setattr(module, _PATCHED_ATTR, True)

    def forward(x: torch.Tensor) -> torch.Tensor:
        weight, bias = module.weight, module.bias
        if is_fp8_tensor(weight):
            base = F.linear(
                x.to(dtype=compute_dtype),
                weight.to(dtype=compute_dtype),
                None if bias is None else bias.to(dtype=compute_dtype),
            )
        else:
            base = F.linear(x, weight, bias)

        for active in getattr(module, _ACTIVE_ATTR):
            low_rank = F.linear(x.to(dtype=compute_dtype), active.down)
            delta = F.linear(low_rank, active.up)
            base = base + delta.to(dtype=base.dtype) * active.multiplier
        return base

    module.forward = forward  # type: ignore[method-assign]
    return True


class RuntimeLinearRegistry:
    def __init__(
        self,
        modules: dict[str, nn.Linear],
        compute_dtype: torch.dtype,
    ):
        self._modules = modules
        self._compute_dtype = compute_dtype
        self._lock = threading.Lock()

    @classmethod
    def patch(
        cls,
        model: nn.Module,
        compute_dtype: torch.dtype,
    ) -> "RuntimeLinearRegistry":
        modules = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        }
        for module in modules.values():
            _patch_linear(module, compute_dtype)
        return cls(modules, compute_dtype)

    def activate(self, prepared: Sequence[PreparedLoRA]) -> "LoRAActivation":
        return LoRAActivation(self, prepared)


class LoRAActivation:
    def __init__(
        self,
        registry: RuntimeLinearRegistry,
        prepared: Sequence[PreparedLoRA],
    ):
        self._registry = registry
        self._prepared = tuple(prepared)
        self._touched: list[nn.Linear] = []
        self._locked = False

    def __enter__(self) -> "LoRAActivation":
        self._registry._lock.acquire()
        self._locked = True
        try:
            grouped: dict[str, list[ActiveLoRALayer]] = {}
            for adapter in self._prepared:
                for layer in adapter.layers:
                    module = self._registry._modules[layer.target]
                    device = module.weight.device
                    grouped.setdefault(layer.target, []).append(
                        ActiveLoRALayer(
                            down=layer.down.to(
                                device=device,
                                dtype=self._registry._compute_dtype,
                            ),
                            up=layer.up.to(
                                device=device,
                                dtype=self._registry._compute_dtype,
                            ),
                            multiplier=adapter.strength * layer.scale,
                        )
                    )
            for target, active in grouped.items():
                module = self._registry._modules[target]
                setattr(module, _ACTIVE_ATTR, tuple(active))
                self._touched.append(module)
            return self
        except Exception:
            self._clear()
            raise

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._clear()
        return False

    def _clear(self) -> None:
        for module in self._touched:
            setattr(module, _ACTIVE_ATTR, ())
        self._touched.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self._locked:
            self._registry._lock.release()
            self._locked = False


class LoRAManager:
    def __init__(
        self,
        model: nn.Module,
        catalog: LoRACatalog,
        compute_dtype: torch.dtype,
    ):
        self.catalog = catalog
        self._loader = LoRALoader(model)
        self._registry = RuntimeLinearRegistry.patch(model, compute_dtype)

    def normalize(self, raw) -> tuple[LoRASelection, ...]:
        return normalize_lora_requests(raw, self.catalog)

    def activation(self, selections: Sequence[LoRASelection]):
        prepared = self._loader.load(selections)
        return self._registry.activate(prepared)
```

The `typing.Sequence` import is intentional. Keep the activation object free of registered `nn.Module` children so temporary LoRA tensors never enter the DiT state dict.

- [ ] **Step 4: Run runtime tests and verify both pass**

Run the Task 3 unittest command. Expected: 2 tests PASS.

- [ ] **Step 5: Commit runtime support**

```bash
git add workers/krea2/krea2_infer/lora_runtime.py workers/krea2/tests/test_lora_runtime.py
git commit -m "feat(krea2): apply LoRA deltas at runtime"
```

### Task 4: Pipeline and denoise-only lifecycle integration

**Files:**
- Modify: `workers/krea2/krea2_infer/pipeline.py`
- Modify: `workers/krea2/krea2_infer/sampling.py`
- Create: `workers/krea2/tests/test_sampling_lora.py`

- [ ] **Step 1: Write a failing sampler lifecycle test**

Create `workers/krea2/tests/test_sampling_lora.py`:

```python
import unittest

import torch
import torch.nn as nn

from krea2_infer.sampling import sample


class RecordingActivation:
    def __init__(self):
        self.active = False
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.active = True
        self.entered += 1
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.active = False
        self.exited += 1
        return False


class FakeModel(nn.Module):
    class Config:
        patch = 1

    def __init__(self, activation):
        super().__init__()
        self.config = self.Config()
        self.activation = activation

    def forward(self, img, context, t, pos, mask):
        if not self.activation.active:
            raise AssertionError("LoRA activation must wrap denoise")
        return torch.zeros_like(img)


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()))

    def forward(self, prompts):
        batch = len(prompts)
        return torch.zeros(batch, 1, 1), torch.ones(batch, 1, dtype=torch.bool)


class FakeAutoencoder:
    compression = 1
    channels = 3

    def __init__(self, activation):
        self.activation = activation

    def decode(self, latent):
        if self.activation.active:
            raise AssertionError("LoRA activation must end before VAE decode")
        return latent.float()


class SamplingLoRALifecycleTests(unittest.TestCase):
    def test_activation_wraps_only_denoise(self):
        activation = RecordingActivation()
        images = sample(
            FakeModel(activation),
            FakeAutoencoder(activation),
            FakeEncoder(),
            ["prompt"],
            device="cpu",
            dtype=torch.float32,
            width=1,
            height=1,
            steps=1,
            guidance=0.0,
            seed=1,
            minres=1,
            maxres=2,
            mu=1.0,
            lora_activation=activation,
        )
        self.assertEqual(len(images), 1)
        self.assertEqual(activation.entered, 1)
        self.assertEqual(activation.exited, 1)
        self.assertFalse(activation.active)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the lifecycle test and verify signature failure**

```bash
PYTHONPATH=workers/krea2 /tmp/runpod-krea2-lora-venv/bin/python -m unittest workers.krea2.tests.test_sampling_lora -v
```

Expected: FAIL because `sample()` does not accept `lora_activation`.

- [ ] **Step 3: Put activation strictly around the Euler loop**

In `sampling.py`, add `from contextlib import nullcontext`. Add the keyword
`lora_activation=None` after `mu=None` in `sample`. Replace lines 165-175 with:

```python
    # Euler integration of the flow ODE with CFG. LoRA tensors live on the
    # DiT device only inside this context and are released before VAE decode.
    img = x
    activation = lora_activation if lora_activation is not None else nullcontext()
    with activation:
        for tcurr, tprev in zip(ts[:-1], ts[1:]):
            t = torch.full((len(img),), tcurr, dtype=img.dtype, device=img.device)
            cond = model(img=img, context=txt, t=t, pos=pos, mask=mask)
            if cfg:
                uncond = model(img=img, context=untxt, t=t, pos=unpos, mask=unmask)
                v = cond + guidance * (cond - uncond)
            else:
                v = cond
            img = img + (tprev - tcurr) * v
```

- [ ] **Step 4: Assemble the manager in `pipeline.py`**

Remove imports `torch.nn as nn` and `torch.nn.functional as F`, the local
`FP8_DTYPES`, `_is_fp8_tensor`, and `_patch_fp8_linears`. Add:

```python
from .lora import LoRACatalog, LoRASelection
from .lora_runtime import LoRAManager, is_fp8_tensor
```

Update FP8 detection in `load_dit` to call `is_fp8_tensor`, and make model
placement independent from patching:

```python
    if has_fp8:
        mmdit = mmdit.to(device=device)
    else:
        mmdit = mmdit.to(device=device, dtype=compute_dtype)
    return mmdit.eval().requires_grad_(False)
```

Add a `loras` field to `Krea2Pipeline`:

```python
    loras: LoRAManager
```

Add `loras: Sequence[LoRASelection] = ()` to `generate`, then before `sample`:

```python
        lora_activation = self.loras.activation(loras) if loras else None
```

Pass `lora_activation=lora_activation` into `sample`.

Add `lora_dir: str | Path | None = None` to `load_pipeline`. Immediately after
normalizing `model_dir`, resolve:

```python
    lora_dir = Path(
        lora_dir
        or os.environ.get("LORA_DIR", "")
        or model_dir / "loras"
    )
```

After `dit = load_dit(...)`, build:

```python
    lora_catalog = LoRACatalog.scan(lora_dir)
    lora_manager = LoRAManager(dit, lora_catalog, dtype)
```

Return:

```python
    return Krea2Pipeline(
        dit=dit,
        ae=ae,
        encoder=encoder,
        device=device,
        loras=lora_manager,
        dtype=dtype,
    )
```

- [ ] **Step 5: Run all unit tests**

Run the full discovery command from Test environment. Expected: catalog,
loader, runtime, and sampler lifecycle tests all PASS.

- [ ] **Step 6: Commit pipeline lifecycle integration**

```bash
git add workers/krea2/krea2_infer/pipeline.py workers/krea2/krea2_infer/sampling.py workers/krea2/tests/test_sampling_lora.py
git commit -m "feat(krea2): activate LoRA only during denoise"
```

### Task 5: RunPod request and response integration

**Files:**
- Modify: `workers/krea2/schemas.py`
- Modify: `workers/krea2/handler.py`
- Modify: `workers/krea2/tests/test_lora.py`

- [ ] **Step 1: Add a failing schema contract test**

Add the import to the import section and insert the test class before the final
`if __name__ == "__main__"` block in `workers/krea2/tests/test_lora.py`:

```python
from schemas import INPUT_SCHEMA


class LoRASchemaTests(unittest.TestCase):
    def test_loras_is_an_optional_list_with_empty_default(self):
        self.assertIs(INPUT_SCHEMA["loras"]["type"], list)
        self.assertFalse(INPUT_SCHEMA["loras"]["required"])
        self.assertEqual(INPUT_SCHEMA["loras"]["default"], [])
```

- [ ] **Step 2: Run the schema test and verify missing key failure**

```bash
PYTHONPATH=workers/krea2 /tmp/runpod-krea2-lora-venv/bin/python -m unittest workers.krea2.tests.test_lora.LoRASchemaTests -v
```

Expected: FAIL with `KeyError: 'loras'`.

- [ ] **Step 3: Extend the RunPod schema**

Add to `INPUT_SCHEMA` in `schemas.py`:

```python
    "loras": {
        "type": list,
        "required": False,
        "default": [],
    },
```

Nested validation remains in `normalize_lora_requests`, because RunPod's flat
validator only establishes that the top-level value is a list.

- [ ] **Step 4: Wire normalized selections through the handler**

Change the import in `handler.py` to:

```python
from krea2_infer import load_pipeline
from krea2_infer.lora import LoRAError
```

At the start of the existing generation `try`, normalize and pass selections:

```python
    try:
        loras = MODELS.pipe.loras.normalize(job_input["loras"])
        images = MODELS.pipe.generate(
            prompt=str(prompt),
            width=width,
            height=height,
            steps=int(job_input["num_inference_steps"]),
            guidance=float(job_input["guidance_scale"]),
            seed=int(seed),
            mu=float(job_input["mu"]) if job_input["mu"] is not None else None,
            num_images=int(job_input["num_images"]),
            negative_prompt=job_input.get("negative_prompt"),
            loras=loras,
        )
    except LoRAError as err:
        logger.warning("Invalid LoRA request: %s", err)
        return {"error": str(err)}
```

Keep the existing OOM, `FileNotFoundError`, and broad exception handlers after
the new `LoRAError` branch. Do not return traceback or `refresh_worker` for
`LoRAError`.

Add normalized applied LoRAs to the success response:

```python
        "loras": [selection.as_dict() for selection in loras],
```

- [ ] **Step 5: Run schema and full unit suites**

Run the single schema command, then full discovery. Expected: all tests PASS.

- [ ] **Step 6: Run syntax checks**

```bash
/tmp/runpod-krea2-lora-venv/bin/python -m py_compile workers/krea2/schemas.py workers/krea2/handler.py workers/krea2/krea2_infer/lora.py workers/krea2/krea2_infer/lora_runtime.py workers/krea2/krea2_infer/pipeline.py workers/krea2/krea2_infer/sampling.py
```

Expected: exit 0 with no output.

- [ ] **Step 7: Commit the API slice**

```bash
git add workers/krea2/schemas.py workers/krea2/handler.py workers/krea2/tests/test_lora.py
git commit -m "feat(krea2): accept per-request LoRA selections"
```

### Task 6: Public deployment documentation

**Files:**
- Modify: `workers/krea2/Dockerfile`
- Modify: `workers/krea2/README.md`

- [ ] **Step 1: Set the default volume directory in Docker**

Extend the existing `ENV MODEL_DIR` block in `Dockerfile`:

```dockerfile
ENV MODEL_DIR=/runpod-volume/krea2 \
    LORA_DIR=/runpod-volume/krea2/loras \
    PYTHONPATH=/app \
    TORCH_COMPILE_DISABLE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

- [ ] **Step 2: Document generic volume and API behavior**

Update `README.md` with all of the following concrete statements and examples:

```text
/runpod-volume/krea2/
  krea2_turbo_fp8.safetensors
  qwen_image_vae.safetensors
  loras/
    lora_a.safetensors
    lora_b.safetensors
```

Add `LORA_DIR` to the env table with default `/runpod-volume/krea2/loras`.
State that the directory is scanned once at worker start, files are loaded only
when selected, and warm workers require restart after directory changes.

Extend the request example with generic IDs only:

```json
"loras": [
  {"name": "lora_a", "strength": 0.8},
  {"name": "lora_b"}
]
```

Document maximum 4, default `1.0`, allowed `0.0..2.0`, exact filename stem as
ID, no URL/path input, standard text-to-image LoRA only, and no `list_loras`
API. Extend the response example with normalized applied values. Remove LoRA
from the `Not in MVP` note while retaining Comfy workflows and baked weights.

Do not add any real LoRA names, trigger words, recommended strengths, download
URLs, or client-specific metadata.

- [ ] **Step 3: Check documentation and Docker diff**

```bash
git diff --check
git diff -- workers/krea2/Dockerfile workers/krea2/README.md
```

Expected: `git diff --check` exits 0; manual diff review shows only generic
directory/API documentation and the `LORA_DIR` env.

- [ ] **Step 4: Commit deployment documentation**

```bash
git add workers/krea2/Dockerfile workers/krea2/README.md
git commit -m "docs(krea2): document runtime LoRA configuration"
```

### Task 7: Final verification and integration handoff

**Files:**
- Verify all files listed in File map
- Do not create GPU smoke scripts or commit actual adapters

- [ ] **Step 1: Run the complete CPU unit suite fresh**

```bash
PYTHONPATH=workers/krea2 /tmp/runpod-krea2-lora-venv/bin/python -m unittest discover -s workers/krea2/tests -p 'test_*.py' -v
```

Expected: all tests PASS with zero failures and zero errors.

- [ ] **Step 2: Compile every worker Python file**

```bash
/tmp/runpod-krea2-lora-venv/bin/python -m compileall -q workers/krea2
```

Expected: exit 0 with no syntax errors.

- [ ] **Step 3: Build the production image**

```bash
docker build -f workers/krea2/Dockerfile -t runpod-krea2:lora-test .
```

Expected: build completes successfully. This verifies packaging and dependency
availability but does not claim GPU inference correctness.

- [ ] **Step 4: Audit repository scope and cleanliness**

```bash
git diff --check
git status --short
git log --oneline -7
rg -n "civitai|trigger word|list_loras|gpu_smoke" workers/krea2 docs/superpowers/plans/2026-07-17-krea2-runtime-lora.md
```

Expected: no whitespace errors; status contains no unintended files; recent
commits match the tasks; search finds only generic explanatory exclusions in
the plan/README and no real adapter metadata or smoke implementation.

- [ ] **Step 5: Prepare the user's manual integration checklist**

Report exactly what remains unverified without a GPU and real private LoRA:

1. Put private `.safetensors` files in `/runpod-volume/krea2/loras/`.
2. Restart/replace warm workers so startup scanning sees them.
3. Send a base job, one-LoRA job, multi-LoRA job, and another base job from the
   private client repository.
4. Confirm the response echoes normalized applied LoRAs and invalid names fail
   without `refresh_worker`.
5. Confirm image effect, acceptable latency, and VRAM behavior manually.

Do not state that GPU inference is verified until the user reports this
integration pass.
