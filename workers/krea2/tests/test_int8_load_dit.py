"""Regression: INT8 weights must load into Linear without requires_grad errors."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file


_MISSING = object()
PACKAGE_ROOT = Path(__file__).parents[1] / "krea2_infer"
_PACKAGE_NAME = "_krea2_int8_load_dit_tests"


def _load_pipeline_modules():
    """Load pipeline + deps under an isolated package (skip heavy TE/VAE)."""
    real = (
        "dit_quant",
        "convrot",
        "int8_linear",
        "lora",
        "lora_runtime",
        "pipeline",
    )
    stubs = ("autoencoder", "encoder", "mmdit", "edit_sampling", "sampling")
    names = (_PACKAGE_NAME, *(f"{_PACKAGE_NAME}.{n}" for n in (*stubs, *real)))
    previous = {n: sys.modules.get(n, _MISSING) for n in names}

    package = types.ModuleType(_PACKAGE_NAME)
    package.__path__ = [str(PACKAGE_ROOT)]
    package.__package__ = _PACKAGE_NAME
    sys.modules[_PACKAGE_NAME] = package

    # Minimal SingleStreamDiT stand-in with one Linear (matches state key "proj.weight").
    mmdit = types.ModuleType(f"{_PACKAGE_NAME}.mmdit")

    class SingleMMDiTConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class SingleStreamDiT(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.proj = nn.Linear(16, 8, bias=True)

    mmdit.SingleMMDiTConfig = SingleMMDiTConfig
    mmdit.SingleStreamDiT = SingleStreamDiT
    sys.modules[mmdit.__name__] = mmdit

    for short, attrs in (
        ("autoencoder", {"QwenAutoencoder": type("QwenAutoencoder", (), {})}),
        (
            "encoder",
            {
                "Qwen3VLConditioner": type("Qwen3VLConditioner", (), {}),
                "TextEncoderConfig": type(
                    "TextEncoderConfig",
                    (),
                    {
                        "__init__": lambda self, model_id: setattr(
                            self, "model_id", model_id
                        ),
                        "max_length": 512,
                        "select_layers": (-1,),
                    },
                ),
            },
        ),
        ("edit_sampling", {"sample_edit": lambda *a, **k: None}),
        ("sampling", {"sample": lambda *a, **k: None}),
    ):
        mod = types.ModuleType(f"{_PACKAGE_NAME}.{short}")
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[mod.__name__] = mod

    try:
        loaded = {}
        for short in real:
            module_name = f"{_PACKAGE_NAME}.{short}"
            spec = importlib.util.spec_from_file_location(
                module_name, PACKAGE_ROOT / f"{short}.py"
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load {short}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            loaded[short] = module
        return loaded["pipeline"], loaded["int8_linear"]
    finally:
        for name, old in previous.items():
            if old is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


_pipeline, _int8 = _load_pipeline_modules()


def _int8_linear_state(prefix: str = "proj") -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    weight_fp = torch.randn(8, 16)
    scale = (weight_fp.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
    q = (weight_fp / scale).round().clamp(-128, 127).to(torch.int8)
    marker = torch.tensor(
        list(
            json.dumps(
                {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": 16,
                }
            ).encode("utf-8")
        ),
        dtype=torch.uint8,
    )
    return {
        f"{prefix}.weight": q,
        f"{prefix}.weight_scale": scale.float(),
        f"{prefix}.comfy_quant": marker,
        f"{prefix}.bias": torch.randn(8),
    }


class LoadStateDictInt8Tests(unittest.TestCase):
    def test_load_state_dict_into_dit_accepts_int8_weights(self):
        """P1: assign=True + int8 + requires_grad=True must not raise."""
        model = _pipeline.SingleStreamDiT(_pipeline.SINGLE_MMDIT_LARGE_WIDE)
        self.assertTrue(model.proj.weight.requires_grad)

        state = _int8_linear_state("proj")
        weight_state, side = _int8.partition_int8_state_dict(state)

        # Must not raise: Only Tensors of floating point ... can require gradients
        _pipeline._load_state_dict_into_dit(
            model, weight_state, Path("fake-int8.safetensors")
        )
        self.assertEqual(model.proj.weight.dtype, torch.int8)
        self.assertFalse(model.proj.weight.requires_grad)

        n = _int8.apply_int8_side_tensors(model, side, weight_state)
        self.assertEqual(n, 1)
        self.assertTrue(model.proj._use_convrot)

    def test_load_dit_int8_roundtrip_from_safetensors(self):
        state = _int8_linear_state("proj")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny_int8.safetensors"
            save_file(state, str(path))
            model = _pipeline.load_dit(
                path,
                device=torch.device("cpu"),
                quant_mode="int8_convrot",
            )
        self.assertEqual(model.proj.weight.dtype, torch.int8)
        self.assertTrue(hasattr(model.proj, "weight_scale"))
        self.assertFalse(model.proj.weight.requires_grad)
        # Forward through patched path is covered elsewhere; ensure materialize flags.
        self.assertTrue(getattr(model.proj, "_is_int8_tensorwise", False))


if __name__ == "__main__":
    unittest.main()
