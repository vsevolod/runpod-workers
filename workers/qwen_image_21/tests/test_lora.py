import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parents[1]))

from qwen_image_21_infer.lora import (
    LoRACatalog,
    LoRAError,
    LoRAManager,
)


class Tensor:
    def __init__(self, *shape):
        self.shape = shape
        self.ndim = len(shape)

    def is_floating_point(self):
        return True


class Pipeline:
    """Diffusers boundary double; no model downloads or GPU required."""

    def __init__(self):
        self.transformer = SimpleNamespace(
            named_modules=lambda: [
                ("linear", SimpleNamespace(in_features=8, out_features=6))
            ]
        )
        self.loaded = set()
        self.active = []
        self.state = {
            "transformer.linear.lora_A.weight": Tensor(2, 8),
            "transformer.linear.lora_B.weight": Tensor(6, 2),
        }
        self.fail_load = False
        self.reads = []

    def lora_state_dict(self, source, **kwargs):
        self.reads.append((source, kwargs))
        return dict(self.state), {"transformer.lora_alpha": 2}

    def load_lora_into_transformer(self, state, **kwargs):
        self.loaded.add(kwargs["adapter_name"])
        self.last_load = kwargs
        if self.fail_load:
            raise ValueError("internal loader details")

    def set_adapters(self, names, adapter_weights):
        assert names, "Cannot activate an empty adapter list on a pristine pipeline"
        assert set(names) <= self.loaded
        self.active = list(zip(names, adapter_weights))

    def get_list_adapters(self):
        return {"transformer": list(self.loaded)}

    def delete_adapters(self, names):
        self.loaded.difference_update(names)
        self.active = [(n, w) for n, w in self.active if n not in names]


class LoRATests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in [
            "style",
            "character",
            "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256",
            "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256",
        ]:
            (self.root / f"{name}.safetensors").write_bytes(b"weights")
        self.catalog = LoRACatalog.scan(self.root)
        self.pipe = Pipeline()
        self.manager = LoRAManager(self.pipe, self.catalog)

    def test_krea_api_defaults_and_zero_strength(self):
        selected = self.catalog.normalize([{"name": "style"}])
        self.assertEqual(
            selected[0].as_dict(), {"name": "style", "strength": 1.0, "type": "lora"}
        )
        self.assertEqual(self.catalog.normalize([{"name": "style", "strength": 0}]), ())
        self.assertEqual(self.catalog.normalize(None), ())
        with self.assertRaises(LoRAError):
            self.catalog.normalize([{"name": "missing", "strength": 0}])

    def test_invalid_requests(self):
        invalid = [
            {},
            ["style"],
            [{"name": "../style"}],
            [{"name": "https://example.com/x"}],
            [{"name": "style.safetensors"}],
            [{"name": "missing"}],
            [{"name": "style", "strength": True}],
            [{"name": "style", "strength": "1"}],
            [{"name": "style", "strength": float("nan")}],
            [{"name": "style", "strength": float("inf")}],
            [{"name": "style", "strength": -0.1}],
            [{"name": "style", "strength": 2.1}],
            [{"name": "style", "type": "weight_diff"}],
            [{"name": "style", "extra": 1}],
            [{"name": "style"}, {"name": "style"}],
            [{"name": "style"}] * 5,
            [{"name": "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256"}],
            [{"name": "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256"}],
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(LoRAError):
                self.catalog.normalize(raw)

    def test_catalog_ignores_symlinks_and_missing_directory(self):
        (self.root / "alias.safetensors").symlink_to(self.root / "style.safetensors")
        catalog = LoRACatalog.scan(self.root)
        with self.assertRaises(LoRAError):
            catalog.normalize([{"name": "alias"}])
        self.assertEqual(LoRACatalog.scan(self.root / "missing").normalize([]), ())

    def test_changed_file_cannot_become_a_symlink(self):
        selected = self.catalog.normalize([{"name": "style"}])
        path = self.root / "style.safetensors"
        path.unlink()
        path.symlink_to(self.root / "character.safetensors")
        with self.assertRaises(LoRAError), self.manager.activate(selected):
            self.fail("symlink must not be loaded")
        self.assertEqual(self.pipe.reads, [])

    def test_multiple_adapters_restore_base_after_success(self):
        selected = self.catalog.normalize(
            [{"name": "style", "strength": 0.8}, {"name": "character", "strength": 0.4}]
        )
        with self.manager.activate(selected):
            self.assertEqual([w for _, w in self.pipe.active], [0.8, 0.4])
            self.assertEqual(len(self.pipe.loaded), 2)
            self.assertIs(self.pipe.last_load["_pipeline"], self.pipe)
            self.assertEqual(
                self.pipe.last_load["metadata"], {"transformer.lora_alpha": 2}
            )
        self.assertEqual(self.pipe.active, [])
        self.assertEqual(self.pipe.loaded, set())
        self.assertTrue(
            all(
                options["local_files_only"] and options["use_safetensors"]
                for _, options in self.pipe.reads
            )
        )
        with self.manager.activate(()):
            self.assertEqual(self.pipe.active, [])

    def test_generation_failure_restores_base(self):
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            with self.manager.activate(self.catalog.normalize([{"name": "style"}])):
                raise RuntimeError("generation failed")
        self.assertEqual(self.pipe.loaded, set())
        self.assertEqual(self.pipe.active, [])

    def test_partial_load_failure_is_safe_and_leaves_no_adapter(self):
        self.pipe.fail_load = True
        with self.assertRaises(LoRAError) as error:
            with self.manager.activate(self.catalog.normalize([{"name": "style"}])):
                self.fail("must not generate")
        self.assertNotIn("internal loader details", str(error.exception))
        self.assertEqual(self.pipe.loaded, set())
        self.assertEqual(self.pipe.active, [])

    def test_failed_attachment_restores_cpu_offload_hooks(self):
        for mode, method in [
            ("cpu", "enable_model_cpu_offload"),
            ("sequential", "enable_sequential_cpu_offload"),
        ]:
            with self.subTest(mode=mode):
                restore = Mock()
                setattr(self.pipe, method, restore)
                manager = LoRAManager(self.pipe, self.catalog, offload=mode)
                self.pipe.fail_load = True
                with (
                    self.assertRaises(LoRAError),
                    manager.activate(self.catalog.normalize([{"name": "style"}])),
                ):
                    self.fail("must not generate")
                restore.assert_called_once_with()

    def test_partial_adapter_is_removed_even_if_diffusers_dropped_its_config(self):
        self.pipe.fail_load = True
        self.pipe.get_list_adapters = lambda: {"transformer": []}
        with (
            self.assertRaises(LoRAError),
            self.manager.activate(self.catalog.normalize([{"name": "style"}])),
        ):
            self.fail("must not generate")
        self.assertEqual(self.pipe.loaded, set())

    def test_wrong_model_and_broken_pairs_rejected_before_injection(self):
        invalid = [
            {},
            {"text_encoder.linear.lora_A.weight": Tensor(2, 8)},
            {
                "transformer.missing.lora_A.weight": Tensor(2, 8),
                "transformer.missing.lora_B.weight": Tensor(6, 2),
            },
            {
                "transformer.linear.lora_A.weight": Tensor(2, 9),
                "transformer.linear.lora_B.weight": Tensor(6, 2),
            },
            {"transformer.linear.lora_A.weight": Tensor(2, 8)},
            {
                "transformer.linear.lora_A.weight": Tensor(2, 8),
                "transformer.linear.lora_B.weight": Tensor(6, 3),
            },
        ]
        for state in invalid:
            with self.subTest(state=state):
                self.pipe.state = state
                with (
                    self.assertRaises(LoRAError),
                    self.manager.activate(self.catalog.normalize([{"name": "style"}])),
                ):
                    self.fail("must not generate")
                self.assertEqual(self.pipe.loaded, set())

    def test_malformed_or_conflicting_fused_mlp_is_rejected_before_injection(self):
        target = "transformer.transformer_blocks.0.img_mlp"
        down = f"{target}.gate_up.lora_A.weight"
        up = f"{target}.gate_up.lora_B.weight"
        for state in (
            {down: Tensor(2, 8)},
            {up: Tensor(12, 2)},
            {down: Tensor(2, 8), up: Tensor(11, 2)},
            {down: Tensor(2, 8), up: Tensor(12, 3)},
            {down: Tensor(2, 8), up: Tensor(12, 2), f"{target}.proj.lora_A.weight": Tensor(2, 8)},
        ):
            with self.subTest(state=state):
                self.pipe.state = state
                with self.assertRaises(LoRAError), self.manager.activate(
                    self.catalog.normalize([{"name": "style"}])
                ):
                    self.fail("invalid fused weights must not generate")
                self.assertEqual(self.pipe.loaded, set())


if __name__ == "__main__":
    unittest.main()
