"""Downloader stays importable without talking to the Hub."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path

_PATH = Path(__file__).parents[1] / "download_weights.py"


def _load():
    name = "_qwen_image_21_download_weights_under_test"
    spec = importlib.util.spec_from_file_location(name, _PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load download_weights")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DownloadWeightsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load()

    def test_model_id(self):
        self.assertEqual(self.mod.MODEL_ID, "Qwen/Qwen-Image-2.1")

    def test_help_exits_cleanly(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                self.mod.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
