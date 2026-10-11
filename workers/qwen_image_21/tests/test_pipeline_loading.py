"""The mmap experiment must reach both loading libraries, not just the pipeline."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1]))
from qwen_image_21_infer import pipeline, runtime_config


class PipelineLoadingTests(unittest.TestCase):
    def test_mmap_experiment_is_enabled_by_default_and_can_be_rolled_back(self):
        with patch.dict(os.environ, {}, clear=True):
            parse = runtime_config.parse_disable_mmap
            self.assertTrue(parse())
        for value in ("0", "false", "no", "off"):
            with patch.dict(os.environ, {"QWEN_DISABLE_MMAP": value}):
                self.assertFalse(parse())
        with self.assertRaises(ValueError):
            parse("invalid")

    def load(self, disable_mmap):
        load = pipeline._load_base_pipeline
        calls = []
        encoder = object()

        class TextEncoder:
            @classmethod
            def from_pretrained(cls, source, **kwargs):
                calls.append(("encoder", source, kwargs))
                return encoder

        class Pipeline:
            @classmethod
            def from_pretrained(cls, source, **kwargs):
                calls.append(("pipeline", source, kwargs))
                return SimpleNamespace(text_encoder=kwargs.get("text_encoder"))

        with patch.dict(
            sys.modules,
            {
                "torch": SimpleNamespace(bfloat16="bf16"),
                "transformers": SimpleNamespace(
                    Qwen3VLForConditionalGeneration=TextEncoder
                ),
            },
        ):
            result = load(
                Pipeline, "/volume/model", local_files_only=True, disable_mmap=disable_mmap
            )
        return result, calls, encoder

    def test_enabled_mode_loads_encoder_eagerly_and_passes_it_into_pipeline(self):
        result, calls, encoder = self.load(True)
        self.assertEqual(
            calls,
            [
                (
                    "encoder",
                    "/volume/model",
                    {
                        "subfolder": "text_encoder",
                        "dtype": "bf16",
                        "local_files_only": True,
                        "disable_mmap": True,
                    },
                ),
                (
                    "pipeline",
                    "/volume/model",
                    {
                        "torch_dtype": "bf16",
                        "local_files_only": True,
                        "disable_mmap": True,
                        "text_encoder": encoder,
                    },
                ),
            ],
        )
        self.assertIs(result.text_encoder, encoder)

    def test_disabled_mode_leaves_encoder_loading_to_pipeline(self):
        _, calls, _ = self.load(False)
        self.assertEqual(
            calls,
            [
                (
                    "pipeline",
                    "/volume/model",
                    {
                        "torch_dtype": "bf16",
                        "local_files_only": True,
                        "disable_mmap": False,
                    },
                )
            ],
        )


HAS_LOADING_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "diffusers", "transformers", "safetensors")
)


@unittest.skipUnless(HAS_LOADING_DEPS, "Install worker dependencies for loading integration tests")
class RealPipelineLoadingTests(unittest.TestCase):
    def test_both_modes_preserve_real_qwen_weights_and_change_loading_backend(self):
        import torch

        if hasattr(torch, "accelerator"):
            try:
                torch.accelerator.current_accelerator()
            except RuntimeError:
                hardware = patch.object(
                    torch.accelerator, "current_accelerator", return_value=None
                )
                hardware.start()
                self.addCleanup(hardware.stop)

        import safetensors.torch
        import transformers.modeling_utils
        from diffusers import DiffusionPipeline, QwenImage21Transformer2DModel
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

        class TinyPipeline(DiffusionPipeline):
            def __init__(self, text_encoder, transformer):
                super().__init__()
                self.register_modules(text_encoder=text_encoder, transformer=transformer)

        config = Qwen3VLConfig(
            text_config=dict(
                vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                head_dim=8,
            ),
            vision_config=dict(
                depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                patch_size=2, temporal_patch_size=1, spatial_merge_size=1,
                out_hidden_size=16, num_position_embeddings=16,
                deepstack_visual_indexes=[],
            ),
        )
        original = TinyPipeline(
            Qwen3VLForConditionalGeneration(config).to(torch.bfloat16),
            QwenImage21Transformer2DModel(
                in_channels=4, out_channels=4, num_layers=1,
                attention_head_dim=8, num_attention_heads=2, context_in_dim=16,
                mlp_ratio=2, axes_dims_rope=(2, 2, 4),
            ).to(torch.bfloat16),
        )
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(directory)
            for disabled in (True, False):
                with self.subTest(disable_mmap=disabled), patch.object(
                    safetensors.torch, "load_file", wraps=safetensors.torch.load_file
                ) as file_loader, patch.object(
                    transformers.modeling_utils, "safe_open",
                    wraps=transformers.modeling_utils.safe_open,
                ) as encoder_loader:
                    restored = pipeline._load_base_pipeline(
                        TinyPipeline, directory,
                        local_files_only=True, disable_mmap=disabled,
                    )
                    self.assertEqual(file_loader.called, not disabled)
                    mmap_calls = [
                        call for call in encoder_loader.call_args_list
                        if call.kwargs.get("backend") == "mmap"
                    ]
                    self.assertEqual(bool(mmap_calls), not disabled)
                    for component in ("text_encoder", "transformer"):
                        actual = getattr(restored, component).state_dict()
                        for name, expected in getattr(original, component).state_dict().items():
                            torch.testing.assert_close(actual[name], expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
