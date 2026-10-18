"""Only completed Krea components may release the corresponding loader."""

import contextlib
import io
import json
import os
import tempfile
import types
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from test_download_quant import _load_download_module


class DownloadPrefetchTests(unittest.TestCase):
    def run_download(self, root, *, fail_encoder=False, fail_adapter=False):
        mod = _load_download_module()
        state = root / "state"
        state.mkdir()
        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = lambda **kwargs: types.SimpleNamespace(model_info=lambda repo: types.SimpleNamespace(sha="a" * 40))

        def file(**kwargs):
            if fail_adapter and kwargs["filename"] == "krea2_general_edit_v2.safetensors":
                raise OSError("V2 adapter transfer failed")
            expected = ("f33b26291c640e8ec50b0da420bfee3c9a660daa"
                        if kwargs["repo_id"] == "conradlocke/krea2-identity-edit" else "a" * 40)
            self.assertEqual(kwargs["revision"], expected)
            path = Path(kwargs["local_dir"]) / kwargs["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"complete dit")
            return str(path)

        def snapshot(repo_id, **kwargs):
            self.assertTrue((state / "dit.json").is_file(), "DiT was not released before later downloads")
            if repo_id != "Qwen/Qwen-Image":
                self.assertTrue((state / "vae.json").is_file(), "VAE must download before the encoder")
                if fail_encoder:
                    raise OSError("encoder transfer failed")
            path = Path(kwargs["cache_dir"]) / repo_id.replace("/", "--") / kwargs["revision"]
            path.mkdir(parents=True)
            (path / "config.json").write_text("{}")
            return str(path)

        hub.hf_hub_download = file
        hub.snapshot_download = snapshot
        with patch.dict(sys.modules, {"huggingface_hub": hub}), patch.dict(os.environ, {
            "MODEL_DIR": str(root / "models"), "HF_HOME": str(root / "hf"), "DIT_QUANT": "int8_convrot",
        }, clear=True), contextlib.redirect_stdout(io.StringIO()):
            return mod.main(["--skip-vae-file", "--prefetch-state-dir", str(state)])

    def test_releases_components_in_loader_order_and_publishes_pinned_local_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(self.run_download(root), 0)
            state = root / "state"
            paths = {name: Path(json.loads((state / f"{name}.json").read_text())["path"])
                     for name in ["dit", "vae", "text_encoder", "all"]}
            self.assertEqual(paths["dit"].read_bytes(), b"complete dit")
            self.assertIn("a" * 40, str(paths["vae"]))
            self.assertIn("a" * 40, str(paths["text_encoder"]))
            self.assertEqual(paths["all"], root / "models")
            adapters = Path(json.loads((state / "edit_loras.json").read_text())["path"])
            self.assertEqual(sorted(path.name for path in adapters.glob("*.safetensors")),
                             ["krea2_general_edit_v2.safetensors", "krea2_identity_edit_v2.safetensors"])

    def test_later_download_failure_never_publishes_all_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(OSError, "encoder transfer failed"):
                self.run_download(root, fail_encoder=True)
            self.assertTrue((root / "state/dit.json").exists())
            self.assertTrue((root / "state/vae.json").exists())
            self.assertFalse((root / "state/all.json").exists())

    def test_partial_v2_download_never_releases_the_edit_catalog_or_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(OSError, "V2 adapter transfer failed"):
                self.run_download(root, fail_adapter=True)
            self.assertTrue((root / "state/text_encoder.json").exists())
            self.assertTrue((root / "models/edit_loras/krea2_identity_edit_v2.safetensors").exists())
            self.assertFalse((root / "state/edit_loras.json").exists())
            self.assertFalse((root / "state/all.json").exists())
