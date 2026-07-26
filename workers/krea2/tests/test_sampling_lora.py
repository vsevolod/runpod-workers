import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn


PACKAGE_ROOT = Path(__file__).parents[1] / "krea2_infer"
_PACKAGE_NAME = "_krea2_sampling_lora_contract_tests"
_MISSING_MODULE = object()


def _load_contract_modules():
    # pipeline / lora_runtime pull int8 + dit_quant; edit_sampling is stubbed.
    real_short_names = (
        "dit_quant",
        "convrot",
        "int8_linear",
        "sampling",
        "lora",
        "lora_runtime",
        "pipeline",
    )
    stub_short_names = (
        "autoencoder",
        "encoder",
        "mmdit",
        "edit_sampling",
    )
    module_names = (
        _PACKAGE_NAME,
        *(f"{_PACKAGE_NAME}.{n}" for n in (*stub_short_names, *real_short_names)),
    )
    previous = {
        name: sys.modules.get(name, _MISSING_MODULE) for name in module_names
    }

    package = types.ModuleType(_PACKAGE_NAME)
    package.__path__ = [str(PACKAGE_ROOT)]
    package.__package__ = _PACKAGE_NAME
    sys.modules[_PACKAGE_NAME] = package

    autoencoder = types.ModuleType(f"{_PACKAGE_NAME}.autoencoder")
    autoencoder.QwenAutoencoder = type("QwenAutoencoder", (), {})
    sys.modules[autoencoder.__name__] = autoencoder

    encoder = types.ModuleType(f"{_PACKAGE_NAME}.encoder")
    encoder.Qwen3VLConditioner = type("Qwen3VLConditioner", (), {})
    encoder.TextEncoderConfig = type(
        "TextEncoderConfig",
        (),
        {
            "__init__": lambda self, model_id: setattr(self, "model_id", model_id),
            "max_length": 512,
            "select_layers": (-1,),
        },
    )
    sys.modules[encoder.__name__] = encoder

    mmdit = types.ModuleType(f"{_PACKAGE_NAME}.mmdit")
    mmdit.SingleMMDiTConfig = type(
        "SingleMMDiTConfig",
        (),
        {"__init__": lambda self, **kwargs: self.__dict__.update(kwargs)},
    )
    mmdit.SingleStreamDiT = type("SingleStreamDiT", (nn.Module,), {})
    sys.modules[mmdit.__name__] = mmdit

    edit_sampling = types.ModuleType(f"{_PACKAGE_NAME}.edit_sampling")
    edit_sampling.sample_edit = lambda *args, **kwargs: None
    sys.modules[edit_sampling.__name__] = edit_sampling

    try:
        loaded = {}
        for short_name in real_short_names:
            module_name = f"{_PACKAGE_NAME}.{short_name}"
            spec = importlib.util.spec_from_file_location(
                module_name,
                PACKAGE_ROOT / f"{short_name}.py",
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load {short_name} contract module")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            loaded[short_name] = module
        return (
            loaded["sampling"],
            loaded["lora"],
            loaded["lora_runtime"],
            loaded["pipeline"],
        )
    finally:
        for name, old_module in previous.items():
            if old_module is _MISSING_MODULE:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


_sampling, _lora, _runtime, _pipeline = _load_contract_modules()


class RecordingActivation:
    def __init__(self, events):
        self.events = events
        self.active = False

    def __enter__(self):
        self.events.append("enter")
        self.active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.active = False
        self.events.append("exit")
        return False


class FakeModel(nn.Module):
    class Config:
        patch = 1

    def __init__(self, events, activation=None):
        super().__init__()
        self.config = self.Config()
        self.events = events
        self.activation = activation

    def forward(self, img, context, t, pos, mask):
        if self.activation is not None and not self.activation.active:
            raise AssertionError("LoRA activation must wrap denoise")
        self.events.append("denoise")
        return torch.zeros_like(img)


class FakeEncoder(nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def forward(self, prompts):
        self.events.append("encode")
        batch = len(prompts)
        return torch.zeros(batch, 1, 2), torch.ones(batch, 1, dtype=torch.bool)


class FakeAutoencoder(nn.Module):
    compression = 1
    channels = 3

    def __init__(self, events, activation=None):
        super().__init__()
        self.events = events
        self.activation = activation

    def decode(self, img):
        if self.activation is not None and self.activation.active:
            raise AssertionError("LoRA activation must end before decode")
        self.events.append("decode")
        return img


class SamplingLoRALifecycleTest(unittest.TestCase):
    def test_activation_starts_after_encoder_offload_and_ends_before_decode(self):
        events = []
        activation = RecordingActivation(events)
        model = FakeModel(events, activation)
        encoder = FakeEncoder(events)
        ae = FakeAutoencoder(events, activation)

        def record_offload(_encoder):
            events.append("offload")

        with mock.patch.object(
            _sampling,
            "_offload_encoder_to_cpu",
            side_effect=record_offload,
        ):
            images = _sampling.sample(
                model,
                ae,
                encoder,
                ["prompt"],
                device="cpu",
                dtype=torch.float32,
                width=1,
                height=1,
                steps=2,
                guidance=0,
                seed=123,
                lora_activation=activation,
            )

        self.assertEqual(len(images), 1)
        self.assertEqual(
            events,
            ["encode", "offload", "enter", "denoise", "denoise", "exit", "decode"],
        )

    def test_sampling_without_activation_preserves_base_path(self):
        events = []
        model = FakeModel(events)
        encoder = FakeEncoder(events)
        ae = FakeAutoencoder(events)

        with mock.patch.object(
            _sampling,
            "_offload_encoder_to_cpu",
            side_effect=lambda _encoder: events.append("offload"),
        ):
            images = _sampling.sample(
                model,
                ae,
                encoder,
                ["prompt"],
                device="cpu",
                dtype=torch.float32,
                width=1,
                height=1,
                steps=1,
                guidance=0,
            )

        self.assertEqual(len(images), 1)
        self.assertEqual(events, ["encode", "offload", "denoise", "decode"])


class PipelineLoRAWiringTest(unittest.TestCase):
    def make_pipeline(self, manager):
        return _pipeline.Krea2Pipeline(
            dit=mock.sentinel.dit,
            ae=mock.sentinel.ae,
            encoder=mock.sentinel.encoder,
            device=torch.device("cpu"),
            loras=manager,
            dtype=torch.float32,
        )

    def test_generate_does_not_activate_manager_for_empty_selection(self):
        manager = mock.Mock()
        pipe = self.make_pipeline(manager)

        with mock.patch.object(_pipeline, "sample", return_value=["image"]) as sampler:
            result = pipe.generate("prompt")

        self.assertEqual(result, ["image"])
        manager.activation.assert_not_called()
        self.assertIsNone(sampler.call_args.kwargs["lora_activation"])

    def test_legacy_positional_construction_preserves_dtype_and_base_generate(self):
        pipe = _pipeline.Krea2Pipeline(
            mock.sentinel.dit,
            mock.sentinel.ae,
            mock.sentinel.encoder,
            torch.device("cpu"),
            torch.float32,
        )

        with mock.patch.object(_pipeline, "sample", return_value=["image"]) as sampler:
            result = pipe.generate("prompt")

        self.assertEqual(result, ["image"])
        self.assertIs(pipe.dtype, torch.float32)
        self.assertIs(sampler.call_args.kwargs["dtype"], torch.float32)
        self.assertIsNone(sampler.call_args.kwargs["lora_activation"])

    def test_legacy_keyword_construction_without_manager_supports_base_generate(self):
        pipe = _pipeline.Krea2Pipeline(
            dit=mock.sentinel.dit,
            ae=mock.sentinel.ae,
            encoder=mock.sentinel.encoder,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        with mock.patch.object(_pipeline, "sample", return_value=["image"]) as sampler:
            result = pipe.generate("prompt")

        self.assertEqual(result, ["image"])
        self.assertIsNone(sampler.call_args.kwargs["lora_activation"])

    def test_nonempty_selection_without_manager_raises_clear_error(self):
        pipe = _pipeline.Krea2Pipeline(
            dit=mock.sentinel.dit,
            ae=mock.sentinel.ae,
            encoder=mock.sentinel.encoder,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        selection = _lora.LoRASelection(
            name="adapter",
            strength=1.0,
            path=Path("/volume/adapter.safetensors"),
        )

        with (
            mock.patch.object(_pipeline, "sample") as sampler,
            self.assertRaisesRegex(RuntimeError, "LoRA manager is not configured"),
        ):
            pipe.generate("prompt", loras=(selection,))

        sampler.assert_not_called()

    def test_generate_passes_requested_activation_to_sampler(self):
        manager = mock.Mock()
        activation = RecordingActivation([])
        manager.activation.return_value = activation
        pipe = self.make_pipeline(manager)
        selection = _lora.LoRASelection(
            name="adapter",
            strength=0.75,
            path=Path("/volume/adapter.safetensors"),
        )

        with mock.patch.object(_pipeline, "sample", return_value=["image"]) as sampler:
            pipe.generate("prompt", loras=(selection,))

        manager.activation.assert_called_once_with((selection,))
        self.assertIs(sampler.call_args.kwargs["lora_activation"], activation)

    def test_load_pipeline_scans_default_lora_directory_once(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / _pipeline.DIT_CANDIDATES[0]).touch()
            (model_dir / _pipeline.VAE_CANDIDATES[0]).touch()
            lora_dir = model_dir / "loras"
            lora_dir.mkdir()
            (lora_dir / "adapter.safetensors").touch()
            dit = nn.Module()
            manager = mock.sentinel.manager

            with (
                mock.patch.dict(os.environ, {"LORA_DIR": ""}, clear=False),
                mock.patch.object(_pipeline, "load_dit", return_value=dit),
                mock.patch.object(_pipeline, "load_autoencoder", return_value=mock.sentinel.ae),
                mock.patch.object(_pipeline, "load_text_encoder", return_value=mock.sentinel.encoder),
                mock.patch.object(_pipeline, "LoRAManager", return_value=manager) as manager_type,
            ):
                pipe = _pipeline.load_pipeline(
                    model_dir=model_dir,
                    device="cpu",
                    dtype=torch.float32,
                    local_files_only=True,
                )

        self.assertIs(pipe.loras, manager)
        manager_type.assert_called_once()
        manager_dit, catalog, manager_dtype = manager_type.call_args.args
        self.assertIs(manager_dit, dit)
        self.assertEqual(catalog.root, lora_dir)
        self.assertEqual(catalog.names, ("adapter",))
        self.assertIs(manager_dtype, torch.float32)

    def test_load_pipeline_prefers_lora_dir_environment_variable(self):
        with (
            tempfile.TemporaryDirectory() as model_directory,
            tempfile.TemporaryDirectory() as lora_directory,
        ):
            model_dir = Path(model_directory)
            lora_dir = Path(lora_directory)
            (model_dir / _pipeline.DIT_CANDIDATES[0]).touch()
            (model_dir / _pipeline.VAE_CANDIDATES[0]).touch()
            (lora_dir / "env-adapter.safetensors").touch()

            with (
                mock.patch.dict(os.environ, {"LORA_DIR": str(lora_dir)}, clear=False),
                mock.patch.object(_pipeline, "load_dit", return_value=nn.Module()),
                mock.patch.object(_pipeline, "load_autoencoder", return_value=mock.sentinel.ae),
                mock.patch.object(_pipeline, "load_text_encoder", return_value=mock.sentinel.encoder),
                mock.patch.object(_pipeline, "LoRAManager") as manager_type,
            ):
                _pipeline.load_pipeline(
                    model_dir=model_dir,
                    device="cpu",
                    dtype=torch.float32,
                    local_files_only=True,
                )

        catalog = manager_type.call_args.args[1]
        self.assertEqual(catalog.root, lora_dir)
        self.assertEqual(catalog.names, ("env-adapter",))


if __name__ == "__main__":
    unittest.main()
