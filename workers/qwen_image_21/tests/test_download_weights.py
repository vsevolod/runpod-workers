"""Downloader stays importable without talking to the Hub."""

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
    def setUp(self):
        env = patch.dict(os.environ, {"QWEN_VAE": "original"})
        env.start()
        self.addCleanup(env.stop)

    @classmethod
    def setUpClass(cls):
        cls.mod = _load()

    def test_model_id(self):
        self.assertEqual(self.mod.MODEL_ID, "Qwen/Qwen-Image-2.1-Turbo")

    def test_prefetch_only_releases_a_validated_snapshot(self):
        for valid in [True, False]:
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                state = root / "state"
                state.mkdir()
                hub = types.ModuleType("huggingface_hub")
                hub.HfApi = lambda **kwargs: types.SimpleNamespace(model_info=lambda repo: types.SimpleNamespace(sha="a" * 40))
                def snapshot(**kwargs):
                    self.assertEqual(kwargs["revision"], "a" * 40)
                    path = Path(kwargs["local_dir"])
                    path.mkdir(parents=True, exist_ok=True)
                    (path / "model_index.json").write_text(json.dumps({
                        "_class_name": "QwenImage21Pipeline",
                        "sample_sigmas": [1, .98, .95, .92, .89, .84, .70, .41] if valid else [],
                    }))
                    return str(path)
                hub.snapshot_download = snapshot
                with patch.dict(sys.modules, {"huggingface_hub": hub}), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    code = self.mod.main(["--output", str(root / "model"), "--prefetch-state-dir", str(state)])
                self.assertEqual(code, 0 if valid else 1)
                self.assertEqual((state / "all.json").exists(), valid)

    def test_downloads_only_turbo_snapshot(self):
        calls = []

        def download(repo_id, local_dir, token, **kwargs):
            calls.append((repo_id, local_dir, token))
            path = Path(local_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "model_index.json").write_text(json.dumps({
                "_class_name": "QwenImage21Pipeline",
                "sample_sigmas": [1, .98, .95, .92, .89, .84, .70, .41],
            }))
            return str(path)

        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = download
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            sys.modules, {"huggingface_hub": hub}
        ), contextlib.redirect_stdout(io.StringIO()):
            code = self.mod.main(["--output", tmp, "--token", "hf_test"])
        self.assertEqual(code, 0)
        self.assertEqual(calls, [("Qwen/Qwen-Image-2.1-Turbo", str(Path(tmp).resolve()), "hf_test")])

    def test_old_snapshot_is_rejected_before_downloading(self):
        hub = types.ModuleType("huggingface_hub")
        def unexpected_download(**kwargs):
            self.fail("An obsolete snapshot must be rejected before contacting the Hub")
        hub.snapshot_download = unexpected_download
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            sys.modules, {"huggingface_hub": hub}
        ), contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            Path(tmp, "model_index.json").write_text("{}")
            self.assertEqual(self.mod.main(["--output", tmp]), 1)

    def test_downloaded_snapshot_must_have_turbo_schedule(self):
        for content in (None, "{}", "not json"):
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                def download(*args):
                    if content is not None:
                        Path(tmp, "model_index.json").write_text(content)
                    return tmp
                hub = types.ModuleType("huggingface_hub")
                hub.snapshot_download = lambda **kwargs: download()
                with patch.dict(sys.modules, {"huggingface_hub": hub}):
                    self.assertEqual(self.mod.main(["--output", tmp]), 1)

    def test_help_exits_cleanly(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                self.mod.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_default_downloads_texture_fix_locally_before_publishing_ready(self):
        calls = []
        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = lambda **kwargs: types.SimpleNamespace(model_info=lambda repo: types.SimpleNamespace(sha="a" * 40))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "state"
            state.mkdir()
            def snapshot(**kwargs):
                self.assertFalse((state / "all.json").exists())
                calls.append(kwargs)
                path = Path(kwargs["local_dir"])
                path.mkdir(parents=True, exist_ok=True)
                if kwargs["repo_id"] == "Qwen/Qwen-Image-2.1-Turbo":
                    (path / "model_index.json").write_text(json.dumps({
                        "_class_name": "QwenImage21Pipeline",
                        "sample_sigmas": [1, .98, .95, .92, .89, .84, .70, .41],
                    }))
                return str(path)
            hub.snapshot_download = snapshot
            with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"huggingface_hub": hub}), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(self.mod.main(["--output", str(root / "model"), "--prefetch-state-dir", str(state)]), 0)
            self.assertEqual(len(calls), 2)
            # Skip the unused stock VAE and the alternative repo's ComfyUI/images files.
            self.assertNotIn("vae/*", calls[0]["allow_patterns"])
            self.assertIn("processor/*", calls[0]["allow_patterns"])
            self.assertEqual(calls[1]["repo_id"], "madebyollin/texture-fix-vae-for-qwen-image-2.1")
            self.assertEqual(calls[1]["local_dir"], str(root / "model" / "texture_fix_vae"))
            self.assertEqual(calls[1]["allow_patterns"], ["config.json", "diffusion_pytorch_model.safetensors"])
            self.assertTrue((state / "all.json").exists())

    def test_texture_fix_download_failure_does_not_publish_ready(self):
        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = lambda **kwargs: types.SimpleNamespace(model_info=lambda repo: types.SimpleNamespace(sha="a" * 40))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "state"
            state.mkdir()
            def snapshot(**kwargs):
                if kwargs["repo_id"] != "Qwen/Qwen-Image-2.1-Turbo":
                    raise OSError("Download interrupted")
                path = Path(kwargs["local_dir"])
                path.mkdir(parents=True, exist_ok=True)
                (path / "model_index.json").write_text(json.dumps({
                    "_class_name": "QwenImage21Pipeline",
                    "sample_sigmas": [1, .98, .95, .92, .89, .84, .70, .41],
                }))
                return str(path)
            hub.snapshot_download = snapshot
            with patch.dict(os.environ, {"QWEN_VAE": "texture-fix"}), patch.dict(sys.modules, {"huggingface_hub": hub}), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(self.mod.main(["--output", str(root / "model"), "--prefetch-state-dir", str(state)]), 1)
            self.assertFalse((state / "all.json").exists())


if __name__ == "__main__":
    unittest.main()
