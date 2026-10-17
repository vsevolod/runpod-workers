"""Exercise external safetensors through the real Diffusers/PEFT loader."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))
HAS_DEPS = all(importlib.util.find_spec(name) is not None for name in (
    "torch", "diffusers", "peft", "safetensors",
))


@unittest.skipUnless(HAS_DEPS, "Install worker dependencies")
class LoRALoadingTests(unittest.TestCase):
    def test_prefixed_external_alpha_and_native_adapters_apply_correct_delta_and_unload(self):
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline, QwenImage21Transformer2DModel
        from safetensors.torch import save_file
        from qwen_image_21_infer.lora import LoRACatalog, LoRAManager

        transformer = QwenImage21Transformer2DModel(
            in_channels=4, out_channels=4, num_layers=1, attention_head_dim=8,
            num_attention_heads=2, context_in_dim=16, mlp_ratio=2, axes_dims_rope=(2, 2, 4),
        )
        class VAE:
            config = SimpleNamespace(z_dim=4)
        class Processor:
            def apply_chat_template(self, *args, **kwargs):
                return [[1]]
            def encode(self, *args, **kwargs):
                return [2]
            @property
            def tokenizer(self):
                return self
        pipe = QwenImage21Pipeline(
            scheduler=FlowMatchEulerDiscreteScheduler(), vae=VAE(),
            processor=Processor(), text_encoder=None, transformer=transformer,
        )
        target = "transformer.transformer_blocks.0.attn.to_q"
        down = torch.full((4, 16), .1)
        up = torch.full((16, 4), .2)
        value = torch.ones(1, 16)
        baseline = transformer.transformer_blocks[0].attn.to_q(value).detach()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for format in ("prefixed_alpha", "down_up_alpha", "native"):
                with self.subTest(format=format):
                    external = format == "down_up_alpha"
                    a, b = ("lora_down", "lora_up") if external else ("lora_A", "lora_B")
                    state = {target + f".{a}.weight": down, target + f".{b}.weight": up}
                    if format != "native":
                        # Match the real Latex adapter's alpha=rank A/B format;
                        # also check alpha/rank scaling for down/up adapters.
                        state[target + ".alpha"] = torch.tensor(2. if external else 4.)
                    save_file(state, root / "style.safetensors")
                    manager = LoRAManager(pipe, LoRACatalog.scan(root), cache_dir=root / "cache")
                    selected = manager.normalize([{"name": "style", "strength": .7}])
                    with manager.activate(selected):
                        actual = transformer.transformer_blocks[0].attn.to_q(value)
                        alpha_scale = .5 if external else 1.
                        expected = baseline + (value @ down.T @ up.T) * alpha_scale * .7
                        torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(transformer.transformer_blocks[0].attn.to_q(value), baseline)
                    self.assertEqual(pipe.get_list_adapters().get("transformer", []), [])


if __name__ == "__main__":
    unittest.main()
