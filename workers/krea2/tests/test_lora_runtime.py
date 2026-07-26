import importlib.util
import sys
import threading
import types
import unittest
import weakref
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F


_MISSING_MODULE = object()
_CANONICAL_PACKAGE_BEFORE = sys.modules.get("krea2_infer", _MISSING_MODULE)
_CANONICAL_LORA_BEFORE = sys.modules.get("krea2_infer.lora", _MISSING_MODULE)
_CANONICAL_RUNTIME_BEFORE = sys.modules.get(
    "krea2_infer.lora_runtime", _MISSING_MODULE
)

PACKAGE_ROOT = Path(__file__).parents[1] / "krea2_infer"
_PACKAGE_NAME = "_krea2_lora_runtime_contract_tests"


def _load_contract_modules():
    # lora_runtime depends on int8_linear → convrot for the INT8 Linear path.
    short_names = ("convrot", "int8_linear", "lora", "lora_runtime")
    module_names = (_PACKAGE_NAME, *(f"{_PACKAGE_NAME}.{n}" for n in short_names))
    previous = {
        name: sys.modules.get(name, _MISSING_MODULE) for name in module_names
    }

    package = types.ModuleType(_PACKAGE_NAME)
    package.__path__ = [str(PACKAGE_ROOT)]
    package.__package__ = _PACKAGE_NAME
    sys.modules[_PACKAGE_NAME] = package

    try:
        loaded = {}
        for short_name in short_names:
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
        return loaded["lora"], loaded["lora_runtime"]
    finally:
        for name, old_module in previous.items():
            if old_module is _MISSING_MODULE:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


_lora, _runtime = _load_contract_modules()
LoRACatalog = _lora.LoRACatalog
LoRALayerWeights = _lora.LoRALayerWeights
LoRASelection = _lora.LoRASelection
PreparedLoRA = _lora.PreparedLoRA
WeightDiffLayerWeights = _lora.WeightDiffLayerWeights
LoRAManager = _runtime.LoRAManager
RuntimeLinearRegistry = _runtime.RuntimeLinearRegistry


class _TwoLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 2, bias=False)
        self.second = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.first(x), self.second(x)


def _prepared(name, strength, *layers, type="lora"):
    return PreparedLoRA(
        name=name, strength=strength, layers=tuple(layers), type=type
    )


def _layer(target, down, up, scale=1.0):
    return LoRALayerWeights(target=target, down=down, up=up, scale=scale)


def _diff_layer(target, delta):
    return WeightDiffLayerWeights(target=target, delta=delta)


class ModuleIsolationTest(unittest.TestCase):
    def test_loader_preserves_canonical_package_bindings(self):
        self.assertIs(
            sys.modules.get("krea2_infer", _MISSING_MODULE),
            _CANONICAL_PACKAGE_BEFORE,
        )
        self.assertIs(
            sys.modules.get("krea2_infer.lora", _MISSING_MODULE),
            _CANONICAL_LORA_BEFORE,
        )
        self.assertIs(
            sys.modules.get("krea2_infer.lora_runtime", _MISSING_MODULE),
            _CANONICAL_RUNTIME_BEFORE,
        )
        self.assertNotIn(_PACKAGE_NAME, sys.modules)


class RuntimeLinearRegistryTest(unittest.TestCase):
    def test_patch_preserves_base_output_and_weight(self):
        model = nn.Module()
        model.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.linear.weight.copy_(torch.eye(2))
        x = torch.tensor([[1.5, -2.0]])
        weight_before = model.linear.weight.detach().clone()

        registry = RuntimeLinearRegistry.patch(model, torch.float32)

        torch.testing.assert_close(model.linear(x), x)
        with registry.activate(()):
            torch.testing.assert_close(model.linear(x), x)
        torch.testing.assert_close(model.linear(x), x)
        torch.testing.assert_close(model.linear.weight, weight_before)

    def test_weight_diff_matches_fused_weight_reference(self):
        model = nn.Module()
        model.linear = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            model.linear.weight.copy_(
                torch.tensor([[1.0, 0.0, -0.5], [0.25, 2.0, 0.0]])
            )
        x = torch.tensor([[1.0, -2.0, 0.5], [0.0, 1.5, -1.0]])
        delta = torch.tensor([[0.0, -0.5, 0.0], [1.0, 0.0, -0.25]])
        strength = 4.0
        adapter = _prepared(
            "fedor",
            strength,
            _diff_layer("linear", delta),
            type="weight_diff",
        )
        fused = model.linear.weight + strength * delta
        expected = F.linear(x, fused)
        registry = RuntimeLinearRegistry.patch(model, torch.float32)

        with registry.activate((adapter,)):
            actual = model.linear(x)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            model.linear.weight,
            fused - strength * delta,
        )

    def test_mix_rank_lora_and_weight_diff_on_different_targets(self):
        model = _TwoLinearModel()
        with torch.no_grad():
            model.first.weight.copy_(torch.eye(2))
            model.second.weight.copy_(2.0 * torch.eye(2))
        x = torch.tensor([[1.0, 3.0]])
        down = torch.tensor([[1.0, 0.0]])
        up = torch.tensor([[2.0], [0.0]])
        delta = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        lora_adapter = _prepared(
            "style",
            0.5,
            _layer("first", down, up, scale=2.0),
            type="lora",
        )
        diff_adapter = _prepared(
            "bypass",
            3.0,
            _diff_layer("second", delta),
            type="weight_diff",
        )
        registry = RuntimeLinearRegistry.patch(model, torch.float32)

        with registry.activate((lora_adapter, diff_adapter)):
            first, second = model(x)

        torch.testing.assert_close(
            first,
            x + 0.5 * 2.0 * F.linear(F.linear(x, down), up),
        )
        torch.testing.assert_close(
            second,
            F.linear(x, model.second.weight + 3.0 * delta),
        )

    def test_weight_diff_activation_clears_after_exit(self):
        model = nn.Module()
        model.linear = nn.Linear(2, 2, bias=False)
        adapter = _prepared(
            "diff",
            1.0,
            _diff_layer("linear", torch.ones(2, 2)),
            type="weight_diff",
        )
        registry = RuntimeLinearRegistry.patch(model, torch.float32)

        with registry.activate((adapter,)):
            self.assertEqual(len(getattr(model.linear, _runtime._ACTIVE_ATTR)), 1)
            self.assertIsNotNone(
                getattr(model.linear, _runtime._ACTIVE_ATTR)[0].delta
            )
        self.assertEqual(getattr(model.linear, _runtime._ACTIVE_ATTR), ())

    def test_two_adapters_on_one_target_sum_their_deltas(self):
        model = nn.Module()
        model.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.linear.weight.copy_(torch.tensor([[1.0, 0.5], [-0.5, 2.0]]))
        x = torch.tensor([[2.0, -1.0], [0.25, 3.0]])
        down_one = torch.tensor([[1.0, 2.0]])
        up_one = torch.tensor([[3.0], [-1.0]])
        down_two = torch.tensor([[2.0, -1.0]])
        up_two = torch.tensor([[0.5], [4.0]])
        adapter_one = _prepared(
            "one",
            0.5,
            _layer("linear", down_one, up_one, scale=2.0),
        )
        adapter_two = _prepared(
            "two",
            1.5,
            _layer("linear", down_two, up_two, scale=0.25),
        )
        base = F.linear(x, model.linear.weight)
        expected = (
            base
            + 0.5 * 2.0 * F.linear(F.linear(x, down_one), up_one)
            + 1.5 * 0.25 * F.linear(F.linear(x, down_two), up_two)
        )
        registry = RuntimeLinearRegistry.patch(model, torch.float32)

        with registry.activate((adapter_one, adapter_two)):
            actual = model.linear(x)

        torch.testing.assert_close(actual, expected)

    def test_non_fp8_linear_keeps_standard_bias_forward(self):
        model = nn.Module()
        model.linear = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            model.linear.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
            model.linear.bias.copy_(torch.tensor([0.25, -0.75]))
        x = torch.tensor([[2.0, -1.0]])
        expected = F.linear(x, model.linear.weight, model.linear.bias)

        RuntimeLinearRegistry.patch(model, torch.float32)

        torch.testing.assert_close(model.linear(x), expected)

    def test_fp8_base_weight_is_cast_only_for_forward(self):
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            self.skipTest("Torch build has no FP8 dtype")
        model = nn.Module()
        model.linear = nn.Linear(2, 2, bias=True)
        weight = torch.tensor([[1.0, 2.0], [-1.0, 0.5]]).to(fp8_dtype)
        bias = torch.tensor([0.5, -0.25]).to(fp8_dtype)
        model.linear.weight = nn.Parameter(weight, requires_grad=False)
        model.linear.bias = nn.Parameter(bias, requires_grad=False)
        x = torch.tensor([[2.0, -1.0]])
        expected = F.linear(x, weight.float(), bias.float())

        RuntimeLinearRegistry.patch(model, torch.float32)

        torch.testing.assert_close(model.linear(x), expected)
        self.assertEqual(model.linear.weight.dtype, fp8_dtype)
        self.assertEqual(model.linear.bias.dtype, fp8_dtype)

    def test_int8_base_weight_uses_int8_path_and_keeps_storage(self):
        model = nn.Module()
        model.linear = nn.Linear(4, 2, bias=True)
        w_fp = torch.tensor(
            [[1.0, -1.0, 0.5, 0.25], [0.5, 1.0, -0.5, 2.0]], dtype=torch.float32
        )
        # Simple per-row quant
        scale = (w_fp.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
        q = (w_fp / scale).round().clamp(-128, 127).to(torch.int8)
        bias = torch.tensor([0.1, -0.2])
        model.linear.weight = nn.Parameter(q, requires_grad=False)
        model.linear.bias = nn.Parameter(bias, requires_grad=False)
        model.linear.register_buffer("weight_scale", scale)
        model.linear._use_convrot = False
        x = torch.tensor([[1.0, 2.0, -1.0, 0.5]])
        expected = F.linear(x, q.float() * scale, bias)

        RuntimeLinearRegistry.patch(model, torch.float32)
        torch.testing.assert_close(model.linear(x), expected, atol=1e-4, rtol=1e-4)
        self.assertEqual(model.linear.weight.dtype, torch.int8)

    def test_activation_cleans_state_after_normal_and_exceptional_exit(self):
        model = nn.Module()
        model.linear = nn.Linear(2, 2, bias=False)
        adapter = _prepared(
            "adapter",
            1.0,
            _layer("linear", torch.ones(1, 2), torch.ones(2, 1)),
        )
        registry = RuntimeLinearRegistry.patch(model, torch.float32)

        with registry.activate((adapter,)):
            self.assertEqual(len(getattr(model.linear, _runtime._ACTIVE_ATTR)), 1)
        self.assertEqual(getattr(model.linear, _runtime._ACTIVE_ATTR), ())

        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            with registry.activate((adapter,)):
                raise RuntimeError("inference failed")
        self.assertEqual(getattr(model.linear, _runtime._ACTIVE_ATTR), ())

        with registry.activate((adapter,)):
            self.assertEqual(len(getattr(model.linear, _runtime._ACTIVE_ATTR)), 1)

    def test_repeated_enter_cleans_state_and_releases_lock(self):
        model = nn.Module()
        model.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.linear.weight.copy_(torch.eye(2))
        x = torch.tensor([[2.0, 3.0]])
        adapter = _prepared(
            "adapter",
            1.0,
            _layer(
                "linear",
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([[1.0], [2.0]]),
            ),
        )
        registry = RuntimeLinearRegistry.patch(model, torch.float32)
        activation = registry.activate((adapter,))
        activation.__enter__()

        try:
            self.assertFalse(torch.equal(model.linear(x), x))
            with self.assertRaisesRegex(RuntimeError, "already active"):
                activation.__enter__()

            torch.testing.assert_close(model.linear(x), x)
            self.assertEqual(getattr(model.linear, _runtime._ACTIVE_ATTR), ())
            acquired = registry._lock.acquire(blocking=False)
            self.assertTrue(acquired)
            if acquired:
                registry._lock.release()

            with registry.activate((adapter,)):
                self.assertFalse(torch.equal(model.linear(x), x))
        finally:
            activation.__exit__(None, None, None)

        activation.__exit__(None, None, None)
        torch.testing.assert_close(model.linear(x), x)

    def test_each_target_receives_only_its_own_layer(self):
        model = _TwoLinearModel()
        with torch.no_grad():
            model.first.weight.copy_(torch.eye(2))
            model.second.weight.copy_(2.0 * torch.eye(2))
        x = torch.tensor([[1.0, 3.0]])
        first_down = torch.tensor([[1.0, 0.0]])
        first_up = torch.tensor([[2.0], [0.0]])
        second_down = torch.tensor([[0.0, 1.0]])
        second_up = torch.tensor([[0.0], [-1.0]])
        adapter = _prepared(
            "both",
            1.0,
            _layer("first", first_down, first_up),
            _layer("second", second_down, second_up),
        )
        registry = RuntimeLinearRegistry.patch(model, torch.float32)

        with registry.activate((adapter,)):
            first, second = model(x)

        torch.testing.assert_close(
            first,
            x + F.linear(F.linear(x, first_down), first_up),
        )
        torch.testing.assert_close(
            second,
            2.0 * x + F.linear(F.linear(x, second_down), second_up),
        )

    def test_activation_does_not_register_lora_state(self):
        model = nn.Module()
        model.linear = nn.Linear(2, 2, bias=False)
        adapter = _prepared(
            "adapter",
            1.0,
            _layer("linear", torch.ones(1, 2), torch.ones(2, 1)),
        )
        state_keys = tuple(model.state_dict())
        parameter_keys = tuple(name for name, _ in model.named_parameters())
        module_keys = tuple(name for name, _ in model.named_modules())
        registry = RuntimeLinearRegistry.patch(model, torch.float32)

        with registry.activate((adapter,)):
            self.assertEqual(tuple(model.state_dict()), state_keys)
            self.assertEqual(
                tuple(name for name, _ in model.named_parameters()), parameter_keys
            )
            self.assertEqual(tuple(name for name, _ in model.named_modules()), module_keys)

        self.assertEqual(tuple(model.state_dict()), state_keys)

    def test_activation_casts_copies_without_mutating_source_tensors(self):
        model = nn.Module()
        model.linear = nn.Linear(2, 2, bias=False)
        down = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        up = torch.tensor([[3.0], [4.0]], dtype=torch.float64)
        down_before = down.clone()
        up_before = up.clone()
        down_pointer = down.data_ptr()
        up_pointer = up.data_ptr()
        adapter = _prepared("adapter", 1.0, _layer("linear", down, up))
        registry = RuntimeLinearRegistry.patch(model, torch.float32)

        with registry.activate((adapter,)):
            active = getattr(model.linear, _runtime._ACTIVE_ATTR)
            self.assertEqual(active[0].down.dtype, torch.float32)
            self.assertEqual(active[0].up.dtype, torch.float32)
            self.assertEqual(active[0].down.device, model.linear.weight.device)
            self.assertEqual(active[0].up.device, model.linear.weight.device)
            self.assertNotEqual(active[0].down.data_ptr(), down_pointer)
            self.assertNotEqual(active[0].up.data_ptr(), up_pointer)

        self.assertEqual(down.dtype, torch.float64)
        self.assertEqual(up.dtype, torch.float64)
        self.assertEqual(down.data_ptr(), down_pointer)
        self.assertEqual(up.data_ptr(), up_pointer)
        torch.testing.assert_close(down, down_before)
        torch.testing.assert_close(up, up_before)

    def test_unknown_target_does_not_poison_the_next_activation(self):
        model = nn.Module()
        model.linear = nn.Linear(2, 2, bias=False)
        unknown = _prepared(
            "bad",
            1.0,
            _layer("linear", torch.ones(1, 2), torch.ones(2, 1)),
            _layer("missing", torch.ones(1, 2), torch.ones(2, 1)),
        )
        valid = _prepared(
            "good",
            1.0,
            _layer("linear", torch.ones(1, 2), torch.ones(2, 1)),
        )
        registry = RuntimeLinearRegistry.patch(model, torch.float32)

        with self.assertRaisesRegex(ValueError, "missing"):
            with registry.activate((unknown,)):
                pass

        self.assertEqual(getattr(model.linear, _runtime._ACTIVE_ATTR), ())
        with registry.activate((valid,)):
            self.assertEqual(len(getattr(model.linear, _runtime._ACTIVE_ATTR)), 1)

    def test_repatching_does_not_double_the_delta(self):
        model = nn.Module()
        model.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.linear.weight.copy_(torch.eye(2))
        x = torch.tensor([[2.0, 3.0]])
        down = torch.tensor([[1.0, 0.0]])
        up = torch.tensor([[1.0], [2.0]])
        adapter = _prepared("adapter", 1.0, _layer("linear", down, up))
        RuntimeLinearRegistry.patch(model, torch.float32)
        registry = RuntimeLinearRegistry.patch(model, torch.float32)
        expected = x + F.linear(F.linear(x, down), up)

        with registry.activate((adapter,)):
            actual = model.linear(x)

        torch.testing.assert_close(actual, expected)

    def test_same_dtype_repatch_returns_the_canonical_registry(self):
        model = nn.Module()
        model.linear = nn.Linear(2, 2, bias=False)

        first = RuntimeLinearRegistry.patch(model, torch.float32)
        second = RuntimeLinearRegistry.patch(model, torch.float32)

        self.assertIs(second, first)
        self.assertIs(second._lock, first._lock)

    def test_repatched_registry_serializes_activation_across_threads(self):
        model = nn.Module()
        model.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.linear.weight.copy_(torch.eye(2))
        x = torch.tensor([[2.0, 3.0]])
        first_adapter = _prepared(
            "first",
            1.0,
            _layer(
                "linear",
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([[1.0], [0.0]]),
            ),
        )
        second_adapter = _prepared(
            "second",
            1.0,
            _layer(
                "linear",
                torch.tensor([[0.0, 1.0]]),
                torch.tensor([[0.0], [1.0]]),
            ),
        )
        expected_first = torch.tensor([[4.0, 3.0]])
        first_registry = RuntimeLinearRegistry.patch(model, torch.float32)
        second_registry = RuntimeLinearRegistry.patch(model, torch.float32)
        attempted = threading.Event()
        entered = threading.Event()
        release_second = threading.Event()
        thread_errors = []

        def activate_second():
            attempted.set()
            try:
                with second_registry.activate((second_adapter,)):
                    entered.set()
                    release_second.wait(timeout=1.0)
            except BaseException as error:
                thread_errors.append(error)

        thread = threading.Thread(target=activate_second, daemon=True)
        try:
            with first_registry.activate((first_adapter,)):
                thread.start()
                self.assertTrue(attempted.wait(timeout=1.0))
                self.assertFalse(entered.wait(timeout=0.1))
                torch.testing.assert_close(model.linear(x), expected_first)

            self.assertTrue(entered.wait(timeout=1.0))
        finally:
            release_second.set()
            thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(thread_errors, [])
        torch.testing.assert_close(model.linear(x), x)

    def test_incompatible_dtype_repatch_preserves_original_registry(self):
        model = nn.Module()
        model.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.linear.weight.copy_(torch.eye(2))
        x = torch.tensor([[2.0, 3.0]])
        adapter = _prepared(
            "adapter",
            1.0,
            _layer(
                "linear",
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([[1.0], [2.0]]),
            ),
        )
        state_keys = tuple(model.state_dict())
        registry = RuntimeLinearRegistry.patch(model, torch.float32)
        forward_before = model.linear.forward

        with self.assertRaisesRegex(ValueError, "compute dtype"):
            RuntimeLinearRegistry.patch(model, torch.float64)

        self.assertIs(model.linear.forward, forward_before)
        self.assertEqual(tuple(model.state_dict()), state_keys)
        self.assertEqual(
            getattr(model.linear, _runtime._PATCHED_ATTR), torch.float32
        )
        with registry.activate((adapter,)):
            torch.testing.assert_close(model.linear(x), torch.tensor([[4.0, 7.0]]))
        torch.testing.assert_close(model.linear(x), x)

    def test_failed_transfer_releases_temporaries_before_empty_cache(self):
        class Converted:
            pass

        class Convertible:
            def __init__(self, references):
                self.references = references

            def to(self, **kwargs):
                converted = Converted()
                self.references.append(weakref.ref(converted))
                return converted

        class FailingTransfer:
            def __init__(self, references):
                self.references = references

            def to(self, **kwargs):
                retained = tuple(reference() for reference in self.references)
                try:
                    raise ValueError("low-level transfer failure")
                except ValueError as cause:
                    if not retained:
                        raise AssertionError("expected converted temporaries")
                    raise RuntimeError("transfer failed") from cause

        model = nn.Module()
        model.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.linear.weight.copy_(torch.eye(2))
        x = torch.tensor([[2.0, 3.0]])
        converted_references = []
        adapter = _prepared(
            "adapter",
            1.0,
            _layer(
                "linear",
                Convertible(converted_references),
                Convertible(converted_references),
            ),
            _layer(
                "linear",
                FailingTransfer(converted_references),
                Convertible(converted_references),
            ),
        )
        registry = RuntimeLinearRegistry.patch(model, torch.float32)

        def assert_temporaries_released():
            self.assertEqual(len(converted_references), 2)
            self.assertTrue(
                all(reference() is None for reference in converted_references)
            )

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                torch.cuda,
                "empty_cache",
                side_effect=assert_temporaries_released,
            ) as empty_cache,
        ):
            with self.assertRaisesRegex(RuntimeError, "transfer failed"):
                with registry.activate((adapter,)):
                    pass

        empty_cache.assert_called_once_with()
        self.assertEqual(getattr(model.linear, _runtime._ACTIVE_ATTR), ())
        torch.testing.assert_close(model.linear(x), x)
        with registry.activate(()):
            torch.testing.assert_close(model.linear(x), x)

    def test_unknown_target_releases_temporaries_before_empty_cache(self):
        class Converted:
            pass

        class Convertible:
            def __init__(self, references):
                self.references = references

            def to(self, **kwargs):
                converted = Converted()
                self.references.append(weakref.ref(converted))
                return converted

        model = nn.Module()
        model.linear = nn.Linear(2, 2, bias=False)
        converted_references = []
        adapter = _prepared(
            "adapter",
            1.0,
            _layer(
                "linear",
                Convertible(converted_references),
                Convertible(converted_references),
            ),
            _layer(
                "missing",
                Convertible(converted_references),
                Convertible(converted_references),
            ),
        )
        registry = RuntimeLinearRegistry.patch(model, torch.float32)

        def assert_temporaries_released():
            self.assertEqual(len(converted_references), 2)
            self.assertTrue(
                all(reference() is None for reference in converted_references)
            )

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                torch.cuda,
                "empty_cache",
                side_effect=assert_temporaries_released,
            ) as empty_cache,
        ):
            with self.assertRaisesRegex(ValueError, "missing"):
                with registry.activate((adapter,)):
                    pass

        empty_cache.assert_called_once_with()
        self.assertEqual(getattr(model.linear, _runtime._ACTIVE_ATTR), ())
        with registry.activate(()):
            self.assertEqual(getattr(model.linear, _runtime._ACTIVE_ATTR), ())


class LoRAManagerTest(unittest.TestCase):
    def test_normalize_delegates_to_contract_normalizer(self):
        model = nn.Module()
        model.linear = nn.Linear(2, 2)
        catalog = object()
        manager = LoRAManager(model, catalog, torch.float32)
        expected = (object(),)

        with mock.patch.object(
            _runtime, "normalize_lora_requests", return_value=expected
        ) as normalize:
            actual = manager.normalize([{"name": "adapter"}])

        self.assertIs(actual, expected)
        normalize.assert_called_once_with([{"name": "adapter"}], catalog)

    def test_activation_loads_only_the_requested_selections_eagerly(self):
        class FakeLoader:
            def __init__(self, prepared):
                self.calls = []
                self.prepared = prepared

            def load(self, selections):
                self.calls.append(selections)
                return self.prepared

        model = nn.Module()
        model.linear = nn.Linear(2, 2, bias=False)
        catalog = LoRACatalog(root=Path("/unused"), _paths={})
        manager = LoRAManager(model, catalog, torch.float32)
        prepared = _prepared(
            "adapter",
            1.0,
            _layer("linear", torch.ones(1, 2), torch.ones(2, 1)),
        )
        fake_loader = FakeLoader((prepared,))
        manager._loader = fake_loader
        selections = (
            LoRASelection(
                name="adapter", strength=1.0, path=Path("/unused/adapter")
            ),
        )

        self.assertEqual(fake_loader.calls, [])
        activation = manager.activation(selections)
        self.assertEqual(fake_loader.calls, [selections])
        with activation:
            self.assertEqual(len(getattr(model.linear, _runtime._ACTIVE_ATTR)), 1)


if __name__ == "__main__":
    unittest.main()
