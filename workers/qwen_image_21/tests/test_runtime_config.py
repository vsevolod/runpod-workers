import importlib.util
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

    def test_missing_snapshot_falls_back_to_hub_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.mod.resolve_model_source(tmp, "Qwen/Qwen-Image-2.1")
        self.assertEqual(source, "Qwen/Qwen-Image-2.1")

    def test_snapshot_with_model_index_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "model_index.json").write_text("{}", encoding="utf-8")
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

    def test_lora_constants_are_the_v021_rank256_adapter(self):
        self.assertEqual(self.mod.LORA_REPO_ID, "Viggle/Qwen-Image-2.1-viggle-turbo")
        self.assertEqual(
            self.mod.LORA_WEIGHT_NAME,
            "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors",
        )

    def test_local_lora_file_wins_over_the_hub(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "loras"
            folder.mkdir()
            weight = folder / self.mod.LORA_WEIGHT_NAME
            weight.write_bytes(b"lora")
            source, name, files_only = self.mod.resolve_lora_source(
                tmp, local_files_only=False
            )
        self.assertEqual(source, str(folder))
        self.assertEqual(name, self.mod.LORA_WEIGHT_NAME)
        self.assertTrue(files_only)

    def test_missing_lora_on_a_local_volume_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                self.mod.resolve_lora_source(tmp, local_files_only=True)

    def test_missing_lora_without_local_only_uses_the_hub(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, name, files_only = self.mod.resolve_lora_source(
                tmp, local_files_only=False
            )
        self.assertEqual(source, "Viggle/Qwen-Image-2.1-viggle-turbo")
        self.assertEqual(name, self.mod.LORA_WEIGHT_NAME)
        self.assertFalse(files_only)


if __name__ == "__main__":
    unittest.main()
