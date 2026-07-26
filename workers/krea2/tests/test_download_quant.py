"""Downloader quant selection must share aliases with resolve_dit_quant."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


DOWNLOAD_PATH = Path(__file__).parents[1] / "download_weights.py"
DIT_QUANT_PATH = Path(__file__).parents[1] / "krea2_infer" / "dit_quant.py"


def _load_download_module():
    # Ensure dit_quant is importable as krea2_infer.dit_quant for download_weights.
    root = Path(__file__).parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    name = "_krea2_download_weights_under_test"
    spec = importlib.util.spec_from_file_location(name, DOWNLOAD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load download_weights")
    module = importlib.util.module_from_spec(spec)
    # Avoid executing main side effects; only need helpers + parse path.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DownloadQuantResolveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_download_module()

    def test_both(self):
        want_fp8, want_int8 = self.mod.resolve_download_quant("both")
        self.assertTrue(want_fp8)
        self.assertTrue(want_int8)

    def test_fp8_aliases(self):
        for raw in ("fp8", "default", "BF16"):
            with self.subTest(raw=raw):
                self.assertEqual(self.mod.resolve_download_quant(raw), (True, False))

    def test_int8_aliases_including_tensorwise(self):
        for raw in ("int8_convrot", "int8", "int8_tensorwise", "INT8_TENSORWISE"):
            with self.subTest(raw=raw):
                self.assertEqual(self.mod.resolve_download_quant(raw), (False, True))

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            self.mod.resolve_download_quant("nvfp4")


if __name__ == "__main__":
    unittest.main()
