"""Shared download behavior, replacing only the external Hub transfers."""

import argparse
import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE = Path(__file__).parents[1] / "model_download.py"


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.is_file(), "The shared download module is missing")
        spec = importlib.util.spec_from_file_location("_worker_model_download_tests", MODULE)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_snapshot_and_file_use_selected_local_paths_cache_and_token(self):
        calls = []
        hub = types.ModuleType("huggingface_hub")

        def snapshot(**kwargs):
            calls.append(kwargs)
            path = Path(kwargs["local_dir"])
            path.mkdir(parents=True, exist_ok=True)
            (path / "model_index.json").write_text("{}")
            return str(path)

        def file(**kwargs):
            calls.append(kwargs)
            path = Path(kwargs["local_dir"]) / kwargs["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"weights")
            return str(path)

        hub.snapshot_download = snapshot
        hub.hf_hub_download = file
        cache = self.root / "hf"
        out = self.root / "models"
        # A stale cache override must not redirect either transfer to a volume.
        os.environ["HF_HUB_CACHE"] = "/old/cache"
        output = io.StringIO()
        with patch.dict(sys.modules, {"huggingface_hub": hub}), contextlib.redirect_stdout(output):
            with self.mod.ModelDownloader(out, hf_home=cache, token="private-test-token") as downloads:
                snapshot_path = downloads.snapshot("example/model", local_dir=out, allow_patterns=["vae/*"])
                file_path = downloads.file("example/model", "weights/model.safetensors")
        self.assertEqual(snapshot_path, out)
        self.assertTrue((out / "model_index.json").is_file())
        self.assertEqual(file_path.read_bytes(), b"weights")
        self.assertEqual(calls, [
            {"repo_id": "example/model", "local_dir": str(out), "allow_patterns": ["vae/*"],
             "cache_dir": str(cache / "hub"), "token": "private-test-token"},
            {"repo_id": "example/model", "filename": "weights/model.safetensors", "local_dir": str(out),
             "cache_dir": str(cache / "hub"), "token": "private-test-token"},
        ])
        self.assertEqual(os.environ["HF_HOME"], str(cache))
        self.assertEqual(os.environ["HF_HUB_CACHE"], str(cache / "hub"))
        self.assertEqual(os.environ["HUGGINGFACE_HUB_CACHE"], str(cache / "hub"))
        self.assertTrue((cache / "hub").is_dir())
        self.assertNotIn("private-test-token", output.getvalue())

    def test_hub_cache_snapshot_does_not_force_a_local_directory(self):
        hub = types.ModuleType("huggingface_hub")
        expected = self.root / "hf/hub/model/snapshot"

        def snapshot(**kwargs):
            self.assertNotIn("local_dir", kwargs)
            self.assertEqual(kwargs["cache_dir"], str(self.root / "hf/hub"))
            expected.mkdir(parents=True)
            return str(expected)

        hub.snapshot_download = snapshot
        with patch.dict(sys.modules, {"huggingface_hub": hub}), contextlib.redirect_stdout(io.StringIO()):
            with self.mod.ModelDownloader(self.root / "models", hf_home=self.root / "hf") as downloads:
                path = downloads.snapshot("example/text-encoder")
        self.assertEqual(path, expected)
        self.assertTrue(path.is_dir())

    def test_failure_is_propagated_without_logging_success(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaisesRegex(ValueError, "bad snapshot"):
            with self.mod.ModelDownloader(self.root / "models"):
                raise ValueError("bad snapshot")
        self.assertIn("stage=model_download failed", output.getvalue())
        self.assertNotIn("stage=model_download done", output.getvalue())

    def test_alias_falls_back_to_copy_when_hardlinks_are_unavailable(self):
        source = self.root / "nested.safetensors"
        source.write_bytes(b"downloaded weights")
        destination = self.root / "flat.safetensors"
        with patch.object(Path, "hardlink_to", side_effect=OSError("unsupported")):
            self.mod.link_or_copy(source, destination)
        self.assertEqual(destination.read_bytes(), b"downloaded weights")

    def test_cli_uses_environment_defaults_and_allows_explicit_overrides(self):
        os.environ.update({"MODEL_DIR": "/models/current", "HF_HOME": "/models/hf",
                           "HUGGING_FACE_HUB_TOKEN": "legacy-token"})
        parser = argparse.ArgumentParser()
        self.mod.add_download_arguments(parser, default_output="./models/default")
        args = parser.parse_args(["--output", "/models/override"])
        self.assertEqual(args.output, "/models/override")
        self.assertEqual(args.hf_home, "/models/hf")
        self.assertEqual(args.token, "legacy-token")


if __name__ == "__main__":
    unittest.main()
