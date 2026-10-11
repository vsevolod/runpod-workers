"""Downloader stays importable without talking to the Hub."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
        self.assertEqual(self.mod.MODEL_ID, "Qwen/Qwen-Image-2.1-Turbo")

    def test_downloads_only_turbo_snapshot(self):
        calls = []

        def download(model_id, local_dir, token):
            calls.append((model_id, local_dir, token))
            path = Path(local_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "model_index.json").write_text(json.dumps({
                "_class_name": "QwenImage21Pipeline",
                "sample_sigmas": [1, .98, .95, .92, .89, .84, .70, .41],
            }))
            return str(path)

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            self.mod, "_snapshot_download", side_effect=download
        ), patch.object(self.mod, "_hf_hub_download", create=True, side_effect=AssertionError("No built-in LoRA download allowed")), contextlib.redirect_stdout(io.StringIO()):
            code = self.mod.main(["--output", tmp, "--token", "hf_test"])
        self.assertEqual(code, 0)
        self.assertEqual(calls, [("Qwen/Qwen-Image-2.1-Turbo", str(Path(tmp).resolve()), "hf_test")])

    def test_old_snapshot_is_rejected_before_downloading(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            self.mod, "_snapshot_download"
        ) as download, contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            Path(tmp, "model_index.json").write_text("{}")
            self.assertEqual(self.mod.main(["--output", tmp]), 1)
            download.assert_not_called()

    def test_downloaded_snapshot_must_have_turbo_schedule(self):
        for content in (None, "{}", "not json"):
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                def download(*args):
                    if content is not None:
                        Path(tmp, "model_index.json").write_text(content)
                    return tmp
                with patch.object(self.mod, "_snapshot_download", side_effect=download) as fetch:
                    self.assertEqual(self.mod.main(["--output", tmp]), 1)
                    fetch.assert_called_once()

    def test_help_exits_cleanly(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                self.mod.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
