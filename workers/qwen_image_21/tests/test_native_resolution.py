"""Exercise the pinned Diffusers pipeline after the same patch used by Docker."""

import importlib.util
import inspect
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[2]))

HAS_DIFFUSERS = all(importlib.util.find_spec(name) is not None for name in ("torch", "diffusers", "transformers"))


def load_patched_pipeline(root):
    from diffusers.pipelines.qwenimage21 import pipeline_qwenimage21

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
    return module


@unittest.skipUnless(HAS_DIFFUSERS, "Install worker dependencies for native-resolution integration tests")
class NativeResolutionTests(unittest.TestCase):
    def test_small_references_denoise_with_real_processor_and_transformer(self):
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Transformer2DModel
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor

        # Keep the checkpoint's actual vision preprocessing, token expansion and
        # transformer grid assembly. Only the expensive encoder/VAE weights are replaced.
        special = ["[UNK]", "[PAD]", "<|image_pad|>", "<|video_pad|>",
                   "<|vision_start|>", "<|vision_end|>", "<|im_start|>", "<|im_end|>"]
        tokenizer = Tokenizer(models.WordLevel({s: i for i, s in enumerate(special)}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        processor = Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessor(
                patch_size=16, merge_size=2, temporal_patch_size=2,
                size={"shortest_edge": 65536, "longest_edge": 16777216},
            ),
            video_processor=Qwen3VLVideoProcessor(),
            tokenizer=PreTrainedTokenizerFast(
                tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]",
                additional_special_tokens=special[2:],
            ),
            chat_template="{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
                          "{% for content in message['content'] %}{{ content['text'] }}{% endfor %}"
                          "<|im_end|>\n{% endfor %}",
        )

        class TextEncoder(torch.nn.Module):
            dtype = torch.float32
            device = torch.device("cpu")

            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.model = SimpleNamespace(norm=torch.nn.Identity())

            def forward(self, input_ids, **kwargs):
                hidden = self.model.norm(torch.zeros(*input_ids.shape, 16, device=input_ids.device))
                return SimpleNamespace(hidden_states=[hidden])

        class VAE:
            config = SimpleNamespace(z_dim=4, latents_mean=[0.] * 4, latents_std=[1.] * 4)

            def encode(self, image):
                return SimpleNamespace(latents=image.new_zeros(1, 4, 1, image.shape[-2] // 16, image.shape[-1] // 16))

        with tempfile.TemporaryDirectory() as tmp:
            module = load_patched_pipeline(Path(tmp))
            transformer = QwenImage21Transformer2DModel(
                in_channels=4, out_channels=4, num_layers=1, attention_head_dim=8,
                num_attention_heads=2, context_in_dim=16, mlp_ratio=2, axes_dims_rope=(2, 2, 4),
            )
            pipe = module.QwenImage21Pipeline(
                scheduler=FlowMatchEulerDiscreteScheduler(), vae=VAE(),
                processor=processor, text_encoder=TextEncoder(), transformer=transformer,
            )
            pipe.set_progress_bar_config(disable=True)
            result = pipe(
                prompt="combine", image=[Image.new("RGB", (96, 32)), Image.new("RGBA", (32, 64))],
                width=256, height=256, reference_resolution=0, true_cfg_scale=1.,
                num_inference_steps=1, output_type="latent",
            )
            self.assertEqual(tuple(result.images.shape), (1, 256, 4))
            self.assertTrue(torch.isfinite(result.images).all())

    def test_native_refs_reach_text_encoder_vae_and_transformer_with_matching_grids(self):
        self._native_reference_case("turbo")

    def test_base_native_refs_reach_transformer_with_forty_unmodified_simple_sigmas(self):
        self._native_reference_case("base")

    def test_five_capped_references_reach_both_encoders_and_transformer(self):
        self._native_reference_case("turbo", capped=True)

    def _native_reference_case(self, model_variant, *, capped=False):
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Transformer2DModel
        from qwen_image_21_infer.sampling import base_scheduler, comfy_simple_sigmas

        with tempfile.TemporaryDirectory() as tmp:
            module = load_patched_pipeline(Path(tmp))

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
                sample_sigmas=[1., .978453, .954180, .926626, .895080, .845148, .704534, .414568] if model_variant == "turbo" else None,
            )
            sampling = {}
            if model_variant == "base":
                pipe.scheduler = base_scheduler(pipe.scheduler)
                sampling = dict(num_inference_steps=40, sigmas=comfy_simple_sigmas(40))
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

            inputs = dict(
                prompt="combine", image=[Image.new("RGB", (96, 64)), Image.new("RGBA", (65, 127))],
                width=64, height=64, output_resolution=1024, reference_resolution=0,
                true_cfg_scale=1., **sampling,
            )
            if capped:
                from shared.api import QWEN_API, parse_job_input
                from shared.api.images import images_to_base64_data_urls
                from qwen_image_21_infer.request import build_pipe_kwargs, normalize_job_input
                parsed = parse_job_input({"input": {
                    "type": "image_edit", "prompt": "combine", "width": 256, "height": 256,
                    "images": images_to_base64_data_urls([Image.new("RGBA", (2048, 2048)) for _ in range(5)]),
                }}, QWEN_API)
                norm = normalize_job_input(parsed.values, raw_keys=parsed.raw_keys)
                inputs = build_pipe_kwargs(norm)
            with patch.object(pipe, "encode_prompt", new=encode_prompt), patch.object(transformer, "forward", new=forward):
                with self.assertRaises(Captured):
                    pipe(**inputs, output_type="latent")
            if capped:
                self.assertEqual(seen["vl_sizes"], [(416, 416)] * 5)
                self.assertEqual(vae.seen, [(1, 4, 1, 416, 416)] * 5)
                self.assertEqual(seen["img_shapes"], [[(1, 26, 26)] * 5 + [(1, 16, 16)]])
                self.assertEqual(seen["latent_shape"], (1, 3636, 4))
                return
            self.assertEqual(seen["vl_sizes"], [(96, 64), (64, 128)])
            self.assertEqual(vae.seen, [(1, 4, 1, 64, 96), (1, 4, 1, 128, 64)])
            self.assertEqual(seen["img_shapes"], [[(1, 4, 6), (1, 8, 4), (1, 4, 4)]])
            self.assertEqual(seen["latent_shape"], (1, 72, 4))
            self.assertEqual(pipe.num_timesteps, 8 if model_variant == "turbo" else 40)
            if model_variant == "base":
                torch.testing.assert_close(pipe.scheduler.sigmas, torch.tensor(comfy_simple_sigmas(40) + [0.]))


if __name__ == "__main__":
    unittest.main()
