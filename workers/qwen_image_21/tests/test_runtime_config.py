import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_PATH = Path(__file__).parents[1] / "qwen_image_21_infer" / "runtime_config.py"


def _load():
    name = "_qwen_image_21_runtime_config_under_test"
    spec = importlib.util.spec_from_file_location(name, _PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load runtime_config")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RuntimeConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load()

    def test_qwen_paths_and_ids_override_generic_settings_from_krea_endpoint(self):
        from unittest.mock import patch
        with patch.dict(os.environ, {
            'MODEL_DIR': '/models/krea2', 'LORA_DIR': '/volume/krea2/loras',
            'MODEL_ID': 'other/model', 'BASE_MODEL_DIR': '/other/base',
            'QWEN_MODEL_DIR': '/models/qwen-turbo', 'QWEN_BASE_MODEL_DIR': '/models/qwen-base',
            'QWEN_LORA_DIR': '/volume/qwen/loras', 'QWEN_MODEL_ID': 'qwen/turbo',
            'QWEN_BASE_MODEL_ID': 'qwen/base',
        }, clear=True):
            self.assertEqual(self.mod.resolve_model_directory('turbo'), Path('/models/qwen-turbo'))
            self.assertEqual(self.mod.resolve_model_directory('base'), Path('/models/qwen-base'))
            self.assertEqual(self.mod.resolve_lora_directory(), Path('/volume/qwen/loras'))
            self.assertEqual(self.mod.resolve_model_id('turbo'), 'qwen/turbo')
            self.assertEqual(self.mod.resolve_model_id('base'), 'qwen/base')

    def test_lora_directory_keeps_generic_and_model_local_fallbacks(self):
        from unittest.mock import patch
        with patch.dict(os.environ, {'LORA_DIR': '/volume/styles'}, clear=True):
            self.assertEqual(self.mod.resolve_lora_directory(), Path('/volume/styles'))
        with patch.dict(os.environ, {'QWEN_MODEL_DIR': '/models/selected'}, clear=True):
            self.assertEqual(self.mod.resolve_lora_directory(), Path('/models/selected/loras'))

    def test_missing_snapshot_falls_back_to_hub_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.mod.resolve_model_source(tmp, "Qwen/Qwen-Image-2.1-Turbo")
        self.assertEqual(source, "Qwen/Qwen-Image-2.1-Turbo")

    def test_snapshot_with_model_index_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "model_index.json").write_text(json.dumps({"_class_name": "QwenImage21Pipeline", "sample_sigmas": [1, .98, .95, .92, .89, .84, .70, .41]}), encoding="utf-8")
            source = self.mod.resolve_model_source(tmp, "ignored/id")
        self.assertEqual(source, tmp)

    def test_env_model_id_when_arguments_omitted(self):
        previous = os.environ.get("MODEL_ID")
        os.environ["MODEL_ID"] = "org/snapshot"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                source = self.mod.resolve_model_source(tmp)
        finally:
            if previous is None:
                os.environ.pop("MODEL_ID", None)
            else:
                os.environ["MODEL_ID"] = previous
        self.assertEqual(source, "org/snapshot")

    def test_offload_aliases(self):
        self.assertEqual(self.mod.resolve_offload("cpu"), "cpu")
        self.assertEqual(self.mod.resolve_offload("MODEL"), "cpu")
        self.assertEqual(self.mod.resolve_offload("sequential"), "sequential")
        self.assertEqual(self.mod.resolve_offload("none"), "none")
        self.assertEqual(self.mod.resolve_offload("cuda"), "none")
        with self.assertRaises(ValueError):
            self.mod.resolve_offload("disk")

    def test_local_files_only(self):
        self.assertTrue(self.mod.parse_local_files_only("1"))
        self.assertTrue(self.mod.parse_local_files_only("yes"))
        self.assertFalse(self.mod.parse_local_files_only(""))
        self.assertFalse(self.mod.parse_local_files_only("0"))

    def test_turbo_defaults_use_a_separate_local_directory(self):
        self.assertEqual(self.mod.DEFAULT_MODEL_ID, "Qwen/Qwen-Image-2.1-Turbo")
        self.assertEqual(self.mod.DEFAULT_MODEL_DIR, "/models/qwen_image_21_turbo")

    def test_base_snapshot_and_paths_cannot_be_mistaken_for_turbo(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            self.assertEqual(str(self.mod.resolve_model_directory("base")), "/models/qwen_image_21_base")
            root = Path(tmp)
            (root / "model_index.json").write_text(json.dumps({"_class_name": "QwenImage21Pipeline"}))
            self.assertEqual(self.mod.resolve_model_source(tmp, model_variant="base"), tmp)
            with self.assertRaisesRegex(ValueError, "Turbo"):
                self.mod.resolve_model_source(tmp, model_variant="turbo")
            (root / "model_index.json").write_text(json.dumps({"_class_name": "QwenImage21Pipeline", "sample_sigmas": [1, .98, .95, .92, .89, .84, .70, .41]}))
            with self.assertRaisesRegex(ValueError, "Base"):
                self.mod.resolve_model_source(tmp, model_variant="base")
            with patch.dict(os.environ, {"MODEL_DIR": tmp, "BASE_MODEL_DIR": tmp}):
                with self.assertRaisesRegex(ValueError, "different directories"):
                    self.mod.resolve_model_directory("base")

    def test_old_or_invalid_local_snapshot_is_rejected(self):
        configs = [
            {}, {"_class_name": "QwenImage21Pipeline"},
            {"_class_name": "OtherPipeline", "sample_sigmas": [1] * 8},
            {"_class_name": "QwenImage21Pipeline", "sample_sigmas": [1] * 8},
            {"_class_name": "QwenImage21Pipeline", "sample_sigmas": [1, .9]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for config in configs:
                with self.subTest(config=config):
                    Path(tmp, "model_index.json").write_text(json.dumps(config))
                    with self.assertRaisesRegex(ValueError, "Turbo"):
                        self.mod.resolve_model_source(tmp)
            Path(tmp, "model_index.json").write_text("invalid json")
            with self.assertRaisesRegex(ValueError, "Turbo"):
                self.mod.resolve_model_source(tmp)


if __name__ == "__main__":
    unittest.main()
