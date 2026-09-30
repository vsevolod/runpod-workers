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
        self.assertEqual(self.mod.LORA_REPO_ID, "Viggle/Qwen-Image-2.1-viggle-turbo")
        self.assertEqual(
            self.mod.LORA_WEIGHT_NAME,
            "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors",
        )

    def test_downloads_base_snapshot_and_turbo_lora(self):
        import tempfile

        calls = []

        def snapshot_download(model_id, local_dir, token):
            calls.append(("snapshot", model_id, local_dir, token))
            path = Path(local_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "model_index.json").write_text("{}", encoding="utf-8")
            return str(path)

        def hf_hub_download(repo_id, filename, local_dir, token):
            calls.append(("lora", repo_id, filename, local_dir, token))
            return str(Path(local_dir) / filename)

        self.mod._snapshot_download = snapshot_download
        self.mod._hf_hub_download = hf_hub_download
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                code = self.mod.main(
                    [
                        "--output",
                        tmp,
                        "--token",
                        "hf_test",
                        "--model-id",
                        "Qwen/Qwen-Image-2.1",
                    ]
                )
        self.assertEqual(code, 0)
        self.assertEqual(calls[0][0], "snapshot")
        self.assertEqual(calls[0][1], "Qwen/Qwen-Image-2.1")
        self.assertEqual(calls[1][0], "lora")
        self.assertEqual(calls[1][1], "Viggle/Qwen-Image-2.1-viggle-turbo")
        self.assertEqual(calls[1][2], self.mod.LORA_WEIGHT_NAME)
        self.assertEqual(Path(calls[1][3]), Path(tmp).resolve() / "loras")
        self.assertEqual(calls[1][4], "hf_test")

    def test_help_exits_cleanly(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                self.mod.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
