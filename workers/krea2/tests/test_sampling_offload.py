"""Exercise both samplers with a simulated CUDA residency budget on CPU."""

import importlib
import sys
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import torch
from PIL import Image


PACKAGE = "_krea2_sampling_offload_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).parents[1] / "krea2_infer")]
sys.modules[PACKAGE] = package
sampling = importlib.import_module(f"{PACKAGE}.sampling")
editing = importlib.import_module(f"{PACKAGE}.edit_sampling")


def cpu_device(value):
    if isinstance(value, (str, torch.device)) and torch.device(value).type == "cuda":
        return "cpu"
    return value


def cpu_tensors():
    """Only replace physical CUDA allocation; sampler control flow stays real."""
    stack = ExitStack()
    for name in ("randn", "zeros", "ones", "arange", "full", "Generator"):
        original = getattr(torch, name)

        def allocate(*args, _original=original, **kwargs):
            if "device" in kwargs:
                kwargs["device"] = cpu_device(kwargs["device"])
            return _original(*args, **kwargs)

        stack.enter_context(mock.patch.object(torch, name, side_effect=allocate))
    original_to = torch.Tensor.to

    def tensor_to(tensor, *args, **kwargs):
        if args:
            args = (cpu_device(args[0]), *args[1:])
        if "device" in kwargs:
            kwargs["device"] = cpu_device(kwargs["device"])
        return original_to(tensor, *args, **kwargs)

    stack.enter_context(mock.patch.object(torch.Tensor, "to", tensor_to))
    stack.enter_context(mock.patch.object(torch.cuda, "empty_cache"))
    return stack


class ResidentModule(torch.nn.Module):
    def __init__(self, name, *, device="cuda:0", dtype=torch.float32):
        super().__init__()
        self.name = name
        self.location = torch.device(device)
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=dtype), requires_grad=False)
        self.register_buffer("scale", torch.ones(1, dtype=torch.float32))
        self.moves = []

    def parameters(self, recurse=True):
        yield types.SimpleNamespace(device=self.location)

    def to(self, device, **kwargs):
        if kwargs:
            raise AssertionError("Device transfer must preserve quantized dtypes")
        self.location = torch.device(device)
        self.moves.append(str(self.location))
        return self


class Encoder(ResidentModule):
    def __init__(self, model, ae, *, fail_at=None, device="cuda:0"):
        super().__init__("encoder", device=device)
        self.model, self.ae = model, ae
        self.fail_at = fail_at
        self.calls = 0

    def to(self, device, **kwargs):
        if torch.device(device).type == "cuda":
            if self.model.location.type != "cpu" or self.ae.location.type != "cpu":
                raise torch.cuda.OutOfMemoryError("DiT/VAE overlap during encoder transfer")
        return super().to(device, **kwargs)

    def forward(self, prompts):
        # This is the production OOM: denoisers still occupy the encoder's VRAM.
        if self.model.location.type != "cpu" or self.ae.location.type != "cpu":
            raise torch.cuda.OutOfMemoryError("DiT/VAE overlap with text encoder")
        self.calls += 1
        if self.calls == self.fail_at:
            raise torch.cuda.OutOfMemoryError("encoding failed")
        return torch.zeros(len(prompts), 2, 3), torch.ones(len(prompts), 2, dtype=torch.bool)

    def grounded_encode(self, text, images, *, grounding_px):
        return self([text])


class Denoiser(ResidentModule):
    config = types.SimpleNamespace(patch=1)

    def __init__(self, *, device="cuda:0"):
        super().__init__("dit", device=device, dtype=torch.int8)
        self.encoder = None
        self.calls = 0

    def forward(self, *, img, **kwargs):
        if self.encoder.location.type != "cpu":
            raise AssertionError("Encoder must leave GPU before denoise")
        if self.location.type != "cuda":
            raise AssertionError("DiT must return to GPU before denoise")
        self.calls += 1
        return torch.zeros_like(img)


class Autoencoder(ResidentModule):
    compression = 1
    channels = 3

    def __init__(self, *, device="cuda:0"):
        super().__init__("vae", device=device)
        self.encoder = None

    def _check_residency(self):
        if self.encoder.location.type != "cpu" or self.location.type != "cuda":
            raise AssertionError("VAE must return only after encoder offload")

    def encode(self, pixels):
        self._check_residency()
        return pixels

    def decode(self, latent):
        self._check_residency()
        return latent


class SamplingOffloadTests(unittest.TestCase):
    def models(self, *, fail_at=None, encoder_device="cuda:0"):
        model, ae = Denoiser(), Autoencoder()
        encoder = Encoder(model, ae, fail_at=fail_at, device=encoder_device)
        model.encoder = ae.encoder = encoder
        return model, ae, encoder

    def run_sampler(self, path, model, ae, encoder, *, guidance=0):
        kwargs = dict(device="cuda:0", dtype=torch.float32, width=16, height=16,
                      steps=1, guidance=guidance, seed=42)
        if path == "generate":
            return sampling.sample(model, ae, encoder, ["one", "two"], **kwargs)
        return editing.sample_edit(
            model, ae, encoder, "edit", [Image.new("RGB", (16, 16))] * 2, **kwargs,
        )

    def test_generate_and_edit_separate_encoder_from_denoisers_on_repeated_jobs(self):
        for path in ("generate", "edit"):
            for encoder_device in ("cpu", "cuda:0"):
                with self.subTest(path=path, encoder_device=encoder_device), cpu_tensors():
                    model, ae, encoder = self.models(encoder_device=encoder_device)
                    for _ in range(2):
                        images = self.run_sampler(path, model, ae, encoder, guidance=2)
                        self.assertEqual(len(images), 2 if path == "generate" else 1)
                        self.assertEqual(encoder.location.type, "cpu")
                        self.assertEqual(model.location, torch.device("cuda:0"))
                        self.assertEqual(ae.location, torch.device("cuda:0"))
                    self.assertEqual(encoder.calls, 4)  # positive + negative per job
                    self.assertEqual(model.weight.dtype, torch.int8)
                    self.assertEqual(model.scale.dtype, torch.float32)

    def test_encoding_errors_offload_encoder_without_reloading_denoisers(self):
        for path in ("generate", "edit"):
            for fail_at in (1, 2):
                with self.subTest(path=path, fail_at=fail_at), cpu_tensors():
                    model, ae, encoder = self.models(fail_at=fail_at)
                    with self.assertRaisesRegex(torch.cuda.OutOfMemoryError, "encoding failed"):
                        self.run_sampler(path, model, ae, encoder, guidance=2)
                    self.assertEqual(encoder.calls, fail_at)
                    for module in (model, ae, encoder):
                        self.assertEqual(module.location.type, "cpu")
                    self.assertEqual(model.calls, 0)
                    # The next request can recover from the all-CPU state.
                    encoder.fail_at = None
                    self.run_sampler(path, model, ae, encoder)

    def test_encoder_transfer_failure_also_leaves_all_models_on_cpu(self):
        with cpu_tensors():
            model, ae, encoder = self.models(encoder_device="cpu")
            original_to = encoder.to

            def fail_transfer(device):
                result = original_to(device)
                if torch.device(device).type == "cuda":
                    raise torch.cuda.OutOfMemoryError("transfer failed")
                return result

            with mock.patch.object(encoder, "to", side_effect=fail_transfer):
                with self.assertRaisesRegex(torch.cuda.OutOfMemoryError, "transfer failed"):
                    self.run_sampler("generate", model, ae, encoder)
            for module in (model, ae, encoder):
                self.assertEqual(module.location.type, "cpu")


if __name__ == "__main__":
    unittest.main()
