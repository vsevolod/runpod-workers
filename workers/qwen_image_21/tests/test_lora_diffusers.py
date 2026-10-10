"""Optional integration tests using real Diffusers/PEFT on a tiny CPU model."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1]))
from qwen_image_21_infer.lora import LoRACatalog, LoRAManager, TURBO_ADAPTER_NAME

HAS_DIFFUSERS = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "diffusers", "peft", "safetensors")
)


@unittest.skipUnless(
    HAS_DIFFUSERS, "Install worker dependencies for Diffusers integration tests"
)
class DiffusersLoRATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        # Torch 2.6 CPU raises here; current Transformers expects None on CPU.
        # Only the test's hardware detection needs this compatibility shim.
        if hasattr(torch, "accelerator"):
            try:
                torch.accelerator.current_accelerator()
            except RuntimeError:
                hardware = patch.object(
                    torch.accelerator, "current_accelerator", return_value=None
                )
                hardware.start()
                cls.addClassCleanup(hardware.stop)

        from diffusers import DiffusionPipeline
        from diffusers.loaders import PeftAdapterMixin, QwenImageLoraLoaderMixin
        from diffusers.models.modeling_utils import ModelMixin

        class Transformer(ModelMixin, PeftAdapterMixin):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 6, bias=False)
                torch.nn.init.zeros_(self.linear.weight)

        class Pipeline(DiffusionPipeline, QwenImageLoraLoaderMixin):
            def __init__(self, transformer):
                super().__init__()
                self.register_modules(transformer=transformer)

        cls.Pipeline = Pipeline
        cls.Transformer = Transformer

    def setUp(self):
        from peft import LoraConfig

        torch = self.torch
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pipe = self.Pipeline(self.Transformer())
        state = {
            "transformer.linear.lora_A.weight": torch.ones(2, 8),
            "transformer.linear.lora_B.weight": torch.full((6, 2), 0.5),
        }
        self.pipe.load_lora_weights(state, adapter_name=TURBO_ADAPTER_NAME)
        self.pipe.set_adapters([TURBO_ADAPTER_NAME], adapter_weights=[1.0])
        config = LoraConfig(r=2, lora_alpha=4, target_modules=["linear"])
        for name, up in [("style", 0.25), ("character", 0.75)]:
            self.pipe.save_lora_weights(
                self.tmp.name,
                weight_name=f"{name}.safetensors",
                transformer_lora_layers={
                    "linear.lora_A.weight": torch.ones(2, 8),
                    "linear.lora_B.weight": torch.full((6, 2), up),
                },
                transformer_lora_adapter_metadata=config.to_dict(),
            )
        self.manager = LoRAManager(self.pipe, LoRACatalog.scan(self.tmp.name))
        self.input = torch.ones(1, 8)

    def assert_output(self, value):
        self.torch.testing.assert_close(
            self.pipe.transformer.linear(self.input), self.torch.full((1, 6), value)
        )

    def test_real_weighted_output_metadata_and_cleanup(self):
        self.assert_output(8.0)
        selected = self.manager.normalize(
            [{"name": "style", "strength": 0.8}, {"name": "character", "strength": 0.4}]
        )
        with self.manager.activate(selected):
            # Metadata alpha/rank=2: 8 + 4*2*0.8 + 12*2*0.4 = 24.
            self.assert_output(24.0)
        self.assert_output(8.0)
        self.assertEqual(
            self.pipe.get_list_adapters(), {"transformer": [TURBO_ADAPTER_NAME]}
        )
        self.assertEqual(set(self.pipe.transformer.linear.lora_A), {TURBO_ADAPTER_NAME})
        with self.manager.activate(
            self.manager.normalize([{"name": "style", "strength": 0.5}])
        ):
            self.assert_output(12.0)
        self.assert_output(8.0)

    def test_real_cleanup_on_generation_failure(self):
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            with self.manager.activate(self.manager.normalize([{"name": "style"}])):
                self.assert_output(16.0)
                raise RuntimeError("generation failed")
        self.assert_output(8.0)
        self.assertEqual(self.pipe.get_active_adapters(), [TURBO_ADAPTER_NAME])


if __name__ == "__main__":
    unittest.main()
