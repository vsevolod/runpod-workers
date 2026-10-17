"""Exercise the pinned Diffusers pipeline after the same patch used by Docker."""

import importlib.util
import inspect
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

HAS_DIFFUSERS = all(importlib.util.find_spec(name) is not None for name in ("torch", "diffusers", "transformers"))


@unittest.skipUnless(HAS_DIFFUSERS, "Install worker dependencies for native-resolution integration tests")
class NativeResolutionTests(unittest.TestCase):
    def test_native_refs_reach_text_encoder_vae_and_transformer_with_matching_grids(self):
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Transformer2DModel
        from diffusers.pipelines.qwenimage21 import pipeline_qwenimage21

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py"
            target.parent.mkdir(parents=True)
            shutil.copyfile(pipeline_qwenimage21.__file__, target)
            patch_file = Path(__file__).parents[1] / "patches/diffusers-native-references.patch"
            if "reference_resolution" not in inspect.signature(pipeline_qwenimage21.QwenImage21Pipeline.__call__).parameters:
                subprocess.run(["git", "apply", str(patch_file)], cwd=root, check=True, capture_output=True)
            name = "diffusers.pipelines.qwenimage21._native_resolution_test"
            spec = importlib.util.spec_from_file_location(name, target)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            class Processor:
                def apply_chat_template(self, *args, **kwargs):
                    return [[1]]

                def encode(self, *args, **kwargs):
                    return [2]

                @property
                def tokenizer(self):
                    return self

            class VAE:
                def __init__(self):
                    self.config = SimpleNamespace(z_dim=4, latents_mean=[0.] * 4, latents_std=[1.] * 4)
                    self.seen = []

                def encode(self, image):
                    self.seen.append(tuple(image.shape))
                    return SimpleNamespace(latents=image.new_zeros(1, 4, 1, image.shape[-2] // 16, image.shape[-1] // 16))

            transformer = QwenImage21Transformer2DModel(
                in_channels=4, out_channels=4, num_layers=1, attention_head_dim=8,
                num_attention_heads=2, context_in_dim=16, mlp_ratio=2, axes_dims_rope=(2, 2, 4),
            )
            vae = VAE()
            pipe = module.QwenImage21Pipeline(
                scheduler=FlowMatchEulerDiscreteScheduler(), vae=vae,
                processor=Processor(), text_encoder=None, transformer=transformer,
                sample_sigmas=[1., .978453, .954180, .926626, .895080, .845148, .704534, .414568],
            )
            pipe.set_progress_bar_config(disable=True)
            seen = {}

            def encode_prompt(**kwargs):
                seen["vl_sizes"] = [image.size for image in kwargs["image"]]
                return torch.zeros(1, 4, 16), None, torch.zeros(1, 4, dtype=torch.bool)

            class Captured(Exception):
                pass

            def forward(**kwargs):
                seen["img_shapes"] = kwargs["img_shapes"]
                seen["latent_shape"] = tuple(kwargs["hidden_states"].shape)
                raise Captured

            with patch.object(pipe, "encode_prompt", new=encode_prompt), patch.object(transformer, "forward", new=forward):
                with self.assertRaises(Captured):
                    pipe(
                        prompt="combine", image=[Image.new("RGB", (96, 64)), Image.new("RGBA", (65, 127))],
                        width=64, height=64, output_resolution=1024,
                        reference_resolution=0, true_cfg_scale=1., output_type="latent",
                    )
            self.assertEqual(seen["vl_sizes"], [(96, 64), (64, 128)])
            self.assertEqual(vae.seen, [(1, 4, 1, 64, 96), (1, 4, 1, 128, 64)])
            self.assertEqual(seen["img_shapes"], [[(1, 4, 6), (1, 8, 4), (1, 4, 4)]])
            self.assertEqual(seen["latent_shape"], (1, 72, 4))
            self.assertEqual(pipe.num_timesteps, 8)


if __name__ == "__main__":
    unittest.main()
