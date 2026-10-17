"""Run the actual pinned Qwen denoising loop with small deterministic components."""

import importlib.util
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1]))
HAS_DEPS = all(importlib.util.find_spec(name) is not None for name in ("torch", "diffusers", "transformers"))


@unittest.skipUnless(HAS_DEPS, "Install worker dependencies")
class SamplingPipelineTests(unittest.TestCase):
    def test_steps_cfg_and_all_samplers_execute_without_reloading_or_leaking_state(self):
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline, QwenImage21Transformer2DModel
        from qwen_image_21_infer.pipeline import QwenImage21Runtime
        from qwen_image_21_infer.lora import LoRACatalog
        from qwen_image_21_infer.request import normalize_job_input

        transformer = QwenImage21Transformer2DModel(
            in_channels=4, out_channels=4, num_layers=1, attention_head_dim=8,
            num_attention_heads=2, context_in_dim=16, mlp_ratio=2, axes_dims_rope=(2, 2, 4),
        )
        template = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, shift_terminal=.02)
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
            scheduler=template, vae=VAE(),
            processor=Processor(), text_encoder=None, transformer=transformer,
            sample_sigmas=[1., .978453, .954180, .926626, .895080, .845148, .704534, .414568],
        )
        pipe.set_progress_bar_config(disable=True)
        encoded, calls, guided = [], [], []

        def encode_prompt(prompt, **kwargs):
            encoded.append(prompt)
            value = 1. if prompt else 0.
            return torch.full((1, 4, 16), value), None, torch.zeros(1, 4, dtype=torch.bool)

        def forward(hidden_states, encoder_hidden_states, **kwargs):
            value = float(encoder_hidden_states[0, 0, 0])
            calls.append(value)
            # The conditional and unconditional branches intentionally differ.
            return (torch.full_like(hidden_states, value * .2 + .1),)

        def capture_step(original):
            def step(output, *args, **kwargs):
                guided.append(output.clone())
                return original(output, *args, **kwargs)
            return step

        # Use the normal runtime call with the real Qwen loop and scheduler;
        # return latent tensors here to avoid requiring a full VAE decoder.
        def invoke(**kwargs):
            original_step = pipe.scheduler.step
            with patch.object(pipe.scheduler, "step", new=capture_step(original_step)):
                out = pipe(output_type="latent", **kwargs)
            return SimpleNamespace(images=[out.images.clone()])

        class Wrapper:
            def __init__(self): self.scheduler = template
            def __call__(self, **kwargs):
                pipe.scheduler = self.scheduler
                return invoke(**kwargs)

        wrapper = Wrapper()
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"LORA_CACHE_DIR": tmp}), patch.object(
            pipe, "encode_prompt", new=encode_prompt
        ), patch.object(transformer, "forward", new=forward):
            runtime = QwenImage21Runtime(wrapper, LoRACatalog({}))
            runtime.loras = SimpleNamespace(activate=lambda selections: nullcontext())
            for sampler in ("euler", "heun", "er_sde", "euler"):
                for steps in (3, 8):
                    for cfg in (1., 3.5):
                        with self.subTest(sampler=sampler, steps=steps, cfg=cfg):
                            payload = {"prompt": "fox", "width": 256, "height": 256,
                                       "num_inference_steps": steps, "guidance_scale": cfg,
                                       "sampler": sampler}
                            norm = normalize_job_input(payload, raw_keys=set(payload))
                            encoded.clear(); calls.clear(); guided.clear()
                            first = runtime.run(norm, seed=1234)[0]
                            evaluations = 2 * steps - 1 if sampler == "heun" else steps
                            self.assertEqual(len(calls), evaluations * (2 if cfg > 1 else 1))
                            self.assertEqual(encoded, ["fox", ""] if cfg > 1 else ["fox"])
                            expected_velocity = .1 + cfg * .2 if cfg > 1 else .3
                            for value in guided:
                                torch.testing.assert_close(value, torch.full_like(value, expected_velocity))
                            self.assertIs(wrapper.scheduler, template)
                            torch.randn(31)  # Global RNG must not affect the request.
                            second = runtime.run(norm, seed=1234)[0]
                            torch.testing.assert_close(first, second, rtol=0, atol=0)
                            self.assertTrue(torch.isfinite(first).all())


if __name__ == "__main__":
    unittest.main()
