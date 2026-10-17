"""Optional integration tests using real Diffusers/PEFT on a tiny CPU model."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1]))
from qwen_image_21_infer.lora import LoRACatalog, LoRAManager

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
        self.manager = LoRAManager(self.pipe, LoRACatalog.scan(self.tmp.name), cache_dir=Path(self.tmp.name) / "local")
        self.input = torch.ones(1, 8)

    def assert_output(self, value):
        self.torch.testing.assert_close(
            self.pipe.transformer.linear(self.input), self.torch.full((1, 6), value)
        )

    def test_real_weighted_output_metadata_and_cleanup(self):
        self.assert_output(0.0)
        selected = self.manager.normalize(
            [{"name": "style", "strength": 0.8}, {"name": "character", "strength": 0.4}]
        )
        with self.manager.activate(selected):
            # Metadata alpha/rank=2: 4*2*0.8 + 12*2*0.4 = 16.
            self.assert_output(16.0)
        self.assert_output(0.0)
        self.assertEqual(
            self.pipe.get_list_adapters().get("transformer", []), []
        )
        self.assertEqual(set(self.pipe.transformer.linear.lora_A), set())
        with self.manager.activate(
            self.manager.normalize([{"name": "style", "strength": 0.5}])
        ):
            self.assert_output(4.0)
        self.assert_output(0.0)

    def test_real_cleanup_on_generation_failure(self):
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            with self.manager.activate(self.manager.normalize([{"name": "style"}])):
                self.assert_output(8.0)
                raise RuntimeError("generation failed")
        self.assert_output(0.0)
        self.assertEqual(self.pipe.get_list_adapters().get("transformer", []), [])

    def test_comfy_fused_qwen21_mlp_preserves_weighted_output_and_cleanup(self):
        import json
        from diffusers import QwenImage21Transformer2DModel
        from peft import LoraConfig
        from safetensors.torch import save_file

        torch = self.torch
        torch.manual_seed(7)
        transformer = QwenImage21Transformer2DModel(
            in_channels=4, out_channels=4, num_layers=1, attention_head_dim=8,
            num_attention_heads=2, context_in_dim=16, mlp_ratio=2,
            axes_dims_rope=(2, 2, 4),
        )
        pipe = self.Pipeline(transformer)
        mlp = transformer.transformer_blocks[0].img_mlp
        x = torch.linspace(-0.5, 0.5, 16).reshape(1, 16)
        baseline = mlp(x).detach().clone()
        a = torch.linspace(0.01, 0.05, 32).reshape(2, 16)
        # Distinct halves expose swapping gate and up, or slicing A instead of B.
        b_gate, b_up = torch.full((32, 2), 0.02), torch.full((32, 2), 0.07)
        out_a, out_b = torch.full((2, 32), 0.03), torch.full((16, 2), 0.04)
        state = {
            "diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_A.weight": a,
            "diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_B.weight": torch.cat([b_gate, b_up]),
            "diffusion_model.transformer_blocks.0.img_mlp.out.lora_A.weight": out_a,
            "diffusion_model.transformer_blocks.0.img_mlp.out.lora_B.weight": out_b,
        }
        for with_metadata, scale, out_scale in ((False, 0.6, 0.6), (True, 1.8, 1.2)):
            with self.subTest(metadata=with_metadata):
                metadata = None
                if with_metadata:
                    config = LoraConfig(
                        r=2, lora_alpha=4, target_modules=["gate_up", "out"],
                        rank_pattern={"img_mlp.gate_up": 2},
                        alpha_pattern={"img_mlp.gate_up": 6},
                    ).to_dict()
                    config["target_modules"] = list(config["target_modules"])
                    metadata = {"lora_adapter_metadata": json.dumps({
                        f"transformer.{key}": value for key, value in config.items()
                    })}
                save_file(state, str(Path(self.tmp.name) / "fused.safetensors"), metadata=metadata)
                manager = LoRAManager(pipe, LoRACatalog.scan(self.tmp.name), cache_dir=Path(self.tmp.name) / "local")
                selected = manager.normalize([{"name": "fused", "strength": 0.6}])
                # Independent fused SwiGLU calculation, including alpha/rank scaling.
                gate = torch.nn.functional.linear(x, mlp.gate_layer.weight) + scale * (x @ a.T @ b_gate.T)
                up = torch.nn.functional.linear(x, mlp.proj.weight) + scale * (x @ a.T @ b_up.T)
                hidden = torch.nn.functional.silu(gate) * up
                expected = torch.nn.functional.linear(hidden, mlp.out.weight) + out_scale * (hidden @ out_a.T @ out_b.T)
                with manager.activate(selected):
                    torch.testing.assert_close(mlp(x), expected)
                torch.testing.assert_close(mlp(x), baseline)
                self.assertEqual(set(mlp.proj.lora_A), set())
                self.assertEqual(pipe.get_list_adapters().get("transformer", []), [])


if __name__ == "__main__":
    unittest.main()
