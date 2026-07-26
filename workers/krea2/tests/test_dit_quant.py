"""Tests for DIT_QUANT mode resolution and DiT candidate selection."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


_MISSING = object()
PACKAGE_ROOT = Path(__file__).parents[1] / "krea2_infer"
_PACKAGE_NAME = "_krea2_dit_quant_tests"


def _load_dit_quant():
    module_names = (_PACKAGE_NAME, f"{_PACKAGE_NAME}.dit_quant")
    previous = {name: sys.modules.get(name, _MISSING) for name in module_names}

    package = types.ModuleType(_PACKAGE_NAME)
    package.__path__ = [str(PACKAGE_ROOT)]
    package.__package__ = _PACKAGE_NAME
    sys.modules[_PACKAGE_NAME] = package

    try:
        module_name = f"{_PACKAGE_NAME}.dit_quant"
        spec = importlib.util.spec_from_file_location(
            module_name, PACKAGE_ROOT / "dit_quant.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load dit_quant")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in previous.items():
            if old is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


_dq = _load_dit_quant()


class ResolveDitQuantTests(unittest.TestCase):
    def test_default_is_fp8(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DIT_QUANT", None)
            self.assertEqual(_dq.resolve_dit_quant(), "fp8")

    def test_accepts_int8_convrot_aliases(self):
        for raw in (
            "int8_convrot",
            "INT8_CONVROT",
            "int8",
            "int8_tensorwise",
            " int8_convrot ",
        ):
            with self.subTest(raw=raw):
                self.assertEqual(_dq.resolve_dit_quant(raw), "int8_convrot")

    def test_accepts_fp8_aliases(self):
        for raw in ("fp8", "FP8", "default", ""):
            with self.subTest(raw=raw):
                # empty string falls back to env, so pass explicit via env for ""
                if raw == "":
                    with mock.patch.dict(os.environ, {"DIT_QUANT": "fp8"}):
                        self.assertEqual(_dq.resolve_dit_quant(""), "fp8")
                else:
                    self.assertEqual(_dq.resolve_dit_quant(raw), "fp8")

    def test_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "Unsupported DIT_QUANT"):
            _dq.resolve_dit_quant("nvfp4")

    def test_reads_env_when_value_none(self):
        with mock.patch.dict(os.environ, {"DIT_QUANT": "int8_convrot"}):
            self.assertEqual(_dq.resolve_dit_quant(None), "int8_convrot")

    def test_candidates_differ_by_mode(self):
        fp8 = _dq.dit_candidates_for("fp8")
        int8 = _dq.dit_candidates_for("int8_convrot")
        self.assertIn("krea2_turbo_fp8.safetensors", fp8)
        self.assertIn("krea2_turbo_int8_convrot.safetensors", int8)
        self.assertNotEqual(fp8, int8)


if __name__ == "__main__":
    unittest.main()
