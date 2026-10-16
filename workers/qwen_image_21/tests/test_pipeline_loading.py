"""The mmap experiment must reach both loading libraries, not just the pipeline."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parents[1]))
from qwen_image_21_infer import pipeline, runtime_config


class PipelineLoadingTests(unittest.TestCase):
    def test_imports_overlap_prefetch_but_pipeline_never_loads_a_partial_snapshot(self):
        sys.path.insert(0, str(Path(__file__).parents[2]))
        from shared.prefetch import PrefetchSession
        schedule = [1, .98, .95, .92, .89, .84, .70, .41]
        pipe = Mock(config={"_class_name": "QwenImage21Pipeline", "sample_sigmas": schedule})
        torch = SimpleNamespace(bfloat16="bf16", cuda=SimpleNamespace(is_available=lambda: True))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "download.py"
            script.write_text('''
import argparse, json, time
from pathlib import Path
from shared.prefetch import ComponentPublisher
p = argparse.ArgumentParser(); p.add_argument('--prefetch-state-dir'); a = p.parse_args()
root = Path(__file__).parent
deadline = time.monotonic() + 5
while not (root / 'imports_done').exists():
    if time.monotonic() > deadline: raise RuntimeError('Imports never overlapped download')
    time.sleep(.01)
(root / 'model_index.json').write_text(json.dumps({'_class_name':'QwenImage21Pipeline','sample_sigmas':[1,.98,.95,.92,.89,.84,.70,.41]}))
(root / 'complete').touch()
ComponentPublisher(a.prefetch_state_dir).ready('all', root)
''')
            def load(*args, **kwargs):
                self.assertTrue((root / "complete").is_file(), "Pipeline read an incomplete snapshot")
                return pipe
            with PrefetchSession([sys.executable, str(script)], env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2])}):
                with patch.dict(sys.modules, {"torch": torch, "diffusers": SimpleNamespace(QwenImage21Pipeline=Mock())}), patch.object(
                    pipeline, "log_runtime_info", side_effect=lambda _: (root / "imports_done").touch(),
                ), patch.object(pipeline, "log_cuda_info"), patch.object(pipeline, "_load_base_pipeline", side_effect=load), patch.dict(os.environ, {}, clear=True):
                    runtime = pipeline.load_pipeline(str(root), offload="none")
            self.assertIs(runtime.pipe, pipe)

    def test_memory_efficient_mmap_is_default_and_eager_loading_is_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            parse = runtime_config.parse_disable_mmap
            self.assertFalse(parse())
        for value in ("1", "true", "yes", "on"):
            with patch.dict(os.environ, {"QWEN_DISABLE_MMAP": value}):
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

    def test_turbo_load_preserves_checkpoint_scheduler_and_has_no_builtin_adapter(self):
        schedule = [1, .98, .95, .92, .89, .84, .70, .41]
        pipe = Mock()
        pipe.config = {"_class_name": "QwenImage21Pipeline", "sample_sigmas": schedule}
        scheduler = pipe.scheduler
        torch = SimpleNamespace(bfloat16="bf16", cuda=SimpleNamespace(is_available=lambda: True))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {
            "torch": torch, "diffusers": SimpleNamespace(QwenImage21Pipeline=Mock()),
        }), patch.object(pipeline, "_load_base_pipeline", return_value=pipe), patch.object(
            pipeline, "log_runtime_info"
        ), patch.object(pipeline, "log_cuda_info"), patch.dict(os.environ, {}, clear=True):
            runtime = pipeline.load_pipeline(tmp, offload="none")
        self.assertIs(runtime.pipe.scheduler, scheduler)
        self.assertEqual(runtime.pipe.config["sample_sigmas"], schedule)
        pipe.load_lora_weights.assert_not_called()
        pipe.set_adapters.assert_not_called()
        pipe.to.assert_called_once_with("cuda")

    def test_hub_pipeline_without_saved_turbo_schedule_is_rejected(self):
        pipe = Mock(config={"_class_name": "QwenImage21Pipeline"})
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {
            "torch": SimpleNamespace(), "diffusers": SimpleNamespace(QwenImage21Pipeline=Mock()),
        }), patch.object(pipeline, "_load_base_pipeline", return_value=pipe), patch.object(
            pipeline, "log_runtime_info"
        ), patch.object(pipeline, "log_cuda_info"), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Turbo"):
                pipeline.load_pipeline(tmp)
        pipe.enable_model_cpu_offload.assert_not_called()


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
