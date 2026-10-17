"""The mmap experiment must reach both loading libraries, not just the pipeline."""

import importlib.util
import json
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
    def test_real_runtime_config_without_serialized_class_name_loads_both_variants(self):
        from diffusers import QwenImage21Pipeline, FlowMatchEulerDiscreteScheduler

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

        for variant in ("turbo", "base"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                real_pipe = QwenImage21Pipeline(
                    scheduler=FlowMatchEulerDiscreteScheduler(), vae=VAE(),
                    processor=Processor(), text_encoder=None, transformer=None,
                    sample_sigmas=[1, .98, .95, .92, .89, .84, .70, .41]
                    if variant == "turbo" else None,
                )
                Path(tmp, "model_index.json").write_text(json.dumps({
                    "_class_name": "QwenImage21Pipeline",
                    "sample_sigmas": [1, .98, .95, .92, .89, .84, .70, .41]
                    if variant == "turbo" else None,
                }))
                with patch.object(pipeline, "_load_base_pipeline", return_value=real_pipe), patch.object(
                    real_pipe, "enable_model_cpu_offload", return_value=None
                ), patch.dict(os.environ, {"QWEN_VAE": "original", "MODEL_DIR": tmp,
                                          "LORA_DIR": tmp, "LORA_CACHE_DIR": tmp}, clear=True):
                    runtime = pipeline.load_pipeline(tmp, offload="cpu", model_variant=variant)
                self.assertIs(runtime.pipe, real_pipe)
                self.assertEqual(runtime.pipe.config.sample_sigmas,
                                 [1, .98, .95, .92, .89, .84, .70, .41]
                                 if variant == "turbo" else None)

    def test_runtime_close_releases_transformer_and_lora_owner_but_keeps_shared_vae(self):
        import gc
        import torch
        import weakref
        from diffusers import DiffusionPipeline
        from qwen_image_21_infer.lora import LoRACatalog

        class TinyPipeline(DiffusionPipeline):
            def __init__(self, transformer, vae):
                super().__init__()
                self.register_modules(transformer=transformer, vae=vae)

        shared_vae = torch.nn.Linear(2, 2)
        pipe = TinyPipeline(torch.nn.Linear(2, 2), shared_vae)
        weights = weakref.ref(pipe.transformer)
        old_pipeline = weakref.ref(pipe)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"LORA_CACHE_DIR": tmp}):
            runtime = pipeline.QwenImage21Runtime(pipe, LoRACatalog({}))
            del pipe
            runtime.close()
        gc.collect()
        self.assertIsNone(weights())
        self.assertIsNone(old_pipeline())
        self.assertIsNone(runtime.pipe)
        self.assertIsNone(runtime.loras)
        self.assertIsNotNone(shared_vae.weight)

    def test_alternative_vae_replaces_stock_weights_and_preserves_schedule(self):
        import torch
        from diffusers import AutoencoderKLQwenImage21, DiffusionPipeline, FlowMatchEulerDiscreteScheduler

        class TinyPipeline(DiffusionPipeline):
            def __init__(self, vae, scheduler, sample_sigmas=None):
                super().__init__()
                self.register_modules(vae=vae, scheduler=scheduler)
                self.register_to_config(sample_sigmas=sample_sigmas)

        def make_vae():
            return AutoencoderKLQwenImage21(
                base_dim=4, decoder_base_dim=4, dim_mult=[1, 1, 1, 1, 1],
                num_res_blocks=1, z_dim=4, latents_mean=[0.] * 4,
                latents_std=[1.] * 4, in_channels=4, out_channels=4,
                is_residual=True, temperal_downsample=[False, True, True, True],
            ).to(torch.bfloat16)

        schedule = [1, .98, .95, .92, .89, .84, .70, .41]
        original = TinyPipeline(make_vae(), FlowMatchEulerDiscreteScheduler(shift=3), schedule)
        alternative = make_vae()
        with torch.no_grad():
            alternative.decoder.conv_out.bias.fill_(0.25)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp, "base")
            vae = Path(tmp, "texture_fix_vae")
            original.save_pretrained(base)
            alternative.save_pretrained(vae)
            # A fresh texture-fix deployment does not download the stock VAE.
            import shutil
            shutil.rmtree(base / "vae")
            restored = pipeline._load_base_pipeline(
                TinyPipeline, str(base), local_files_only=True,
                disable_mmap=False, vae_source=str(vae),
            )
            torch.testing.assert_close(restored.vae.decoder.conv_out.bias, torch.full_like(restored.vae.decoder.conv_out.bias, 0.25))
            self.assertEqual(restored.config.sample_sigmas, schedule)
            self.assertEqual(restored.scheduler.config.shift, 3)
            with patch.object(AutoencoderKLQwenImage21, "from_pretrained", side_effect=AssertionError("Shared VAE must not be loaded twice")):
                second = pipeline._load_base_pipeline(
                    TinyPipeline, str(base), local_files_only=True, disable_mmap=False,
                    vae_source="unused", vae=restored.vae,
                )
            self.assertIs(second.vae, restored.vae)

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
