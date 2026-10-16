"""Downloader quant selection must share aliases with resolve_dit_quant."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


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

    def test_prepares_selected_dit_encoder_and_clean_vae_on_local_disk(self):
        # Replace only the Hub transfers; run the downloader's path selection,
        # cache setup and INT8 hardlink against a real temporary filesystem.
        hub = types.ModuleType("huggingface_hub")

        def download(*, repo_id, filename, local_dir, token, cache_dir):
            self.assertEqual(repo_id, "Comfy-Org/Krea-2")
            path = Path(local_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test weights")
            return str(path)

        def snapshot(repo_id, **kwargs):
            cache = Path(os.environ["HF_HOME"]) / "hub"
            path = cache / ("models--" + repo_id.replace("/", "--")) / "snapshots" / "revision"
            if repo_id == "Qwen/Qwen-Image":
                self.assertEqual(kwargs["allow_patterns"], ["vae/*", "VAE/*", "*.json"])
                path /= "vae"
            path.mkdir(parents=True, exist_ok=True)
            (path / "config.json").write_text(json.dumps({"repo": repo_id}))
            return str(path)

        hub.hf_hub_download = download
        hub.snapshot_download = snapshot
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "models"
            cache = root / "hf"
            with patch.dict(sys.modules, {"huggingface_hub": hub}), patch.dict(
                os.environ,
                {"MODEL_DIR": str(model), "HF_HOME": str(cache),
                 "DIT_QUANT": "int8_convrot", "TEXT_ENCODER_ID": "example/conditioner"},
            ), contextlib.redirect_stdout(io.StringIO()):
                code = self.mod.main(["--skip-vae-file"])
            self.assertEqual(code, 0)
            self.assertEqual((model / "krea2_turbo_int8_convrot.safetensors").read_bytes(), b"test weights")
            self.assertFalse((model / "krea2_turbo_fp8.safetensors").exists())
            self.assertFalse((model / "qwen_image_vae.safetensors").exists())
            self.assertTrue((cache / "hub/models--example--conditioner/snapshots/revision/config.json").is_file())
            self.assertTrue((cache / "hub/models--Qwen--Qwen-Image/snapshots/revision/vae/config.json").is_file())

    def test_existing_local_encoder_is_not_sent_to_hub_as_a_repo_id(self):
        hub = types.ModuleType("huggingface_hub")
        with tempfile.TemporaryDirectory() as tmp:
            encoder = Path(tmp) / "encoder"
            encoder.mkdir()
            (encoder / "config.json").write_text("{}")
            vae = Path(tmp) / "hf/vae"

            def snapshot(repo_id, **kwargs):
                self.assertEqual(repo_id, "Qwen/Qwen-Image")
                vae.mkdir(parents=True)
                (vae / "config.json").write_text("{}")
                return str(vae)

            hub.snapshot_download = snapshot
            with patch.dict(sys.modules, {"huggingface_hub": hub}), patch.dict(
                os.environ, {"TEXT_ENCODER_ID": str(encoder)},
            ), contextlib.redirect_stdout(io.StringIO()):
                code = self.mod.main([
                    "--output", str(Path(tmp) / "models"), "--hf-home", str(Path(tmp) / "hf"),
                    "--skip-dit", "--skip-vae-file",
                ])
            self.assertEqual(code, 0)
            self.assertTrue((vae / "config.json").is_file())


if __name__ == "__main__":
    unittest.main()
