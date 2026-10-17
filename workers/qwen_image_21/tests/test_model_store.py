import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parents[1]))


class ModelStoreTests(unittest.TestCase):
    def store(self):
        try:
            return importlib.import_module("qwen_image_21_infer.model_store")
        except ImportError:
            self.fail("Complete local snapshot checking is not implemented")

    def test_manifest_detects_missing_truncated_and_wrong_variant_weights(self):
        store = self.store()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "transformer").mkdir()
            weights = root / "transformer/model.safetensors"
            weights.write_bytes(b"complete weights")
            (root / "model_index.json").write_text(json.dumps({"_class_name": "QwenImage21Pipeline"}))
            options = dict(model_id="Qwen/Qwen-Image-2.1", model_variant="base", profile="texture-fix")
            store.write_manifest(root, **options)
            self.assertTrue(store.snapshot_ready(root, **options))
            self.assertFalse(store.snapshot_ready(root, **{**options, "model_variant": "turbo"}))
            self.assertFalse(store.snapshot_ready(root, **{**options, "profile": "original"}))
            weights.write_bytes(b"partial")
            self.assertFalse(store.snapshot_ready(root, **options))
            store.remove_incomplete_files(root)
            self.assertFalse(weights.exists(), "Remove incomplete files so Hub metadata cannot mistake them for cached weights")
            weights.write_bytes(b"complete weights")
            self.assertTrue(store.snapshot_ready(root, **options))
            weights.unlink()
            self.assertFalse(store.snapshot_ready(root, **options))

    def test_first_base_prefetch_selects_only_base_and_shared_vae(self):
        store = self.store()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "MODEL_DIR": str(Path(tmp) / "turbo"), "BASE_MODEL_DIR": str(Path(tmp) / "base"),
            "QWEN_VAE_DIR": str(Path(tmp) / "vae"), "QWEN_VAE": "texture-fix",
        }), patch.object(store, "PrefetchSession") as session:
            with store.prefetch_model("base") as pending:
                self.assertIs(pending, session.return_value.__enter__.return_value)
            command = session.call_args.args[0]
            self.assertIn("--model-variant", command)
            self.assertEqual(command[command.index("--model-variant") + 1], "base")
            self.assertEqual(command[command.index("--output") + 1], str(Path(tmp) / "base"))
            self.assertEqual(command[command.index("--vae-dir") + 1], str(Path(tmp) / "vae"))
            self.assertFalse((Path(tmp) / "turbo").exists())

    def test_complete_warm_snapshot_does_not_start_a_downloader(self):
        store = self.store()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "MODEL_DIR": str(Path(tmp) / "turbo"), "QWEN_VAE_DIR": str(Path(tmp) / "vae"),
            "QWEN_VAE": "texture-fix",
        }):
            root, vae = Path(tmp) / "turbo", Path(tmp) / "vae"
            for path in (root, vae):
                path.mkdir()
                (path / "config.json").write_text("{}")
            store.write_manifest(root, model_id="Qwen/Qwen-Image-2.1-Turbo", model_variant="turbo", profile="texture-fix")
            store.write_manifest(vae, model_id="madebyollin/texture-fix-vae-for-qwen-image-2.1", model_variant="vae")
            with patch.object(store, "PrefetchSession") as session:
                with store.prefetch_model("turbo") as pending:
                    self.assertIsNone(pending)
                session.assert_not_called()
                (vae / "config.json").unlink()
                with store.prefetch_model("turbo"):
                    pass
                session.assert_called_once()


if __name__ == "__main__":
    unittest.main()
