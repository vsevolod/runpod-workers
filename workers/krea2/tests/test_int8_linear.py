"""Tests for INT8 tensorwise linear + ConvRot helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


_MISSING = object()
PACKAGE_ROOT = Path(__file__).parents[1] / "krea2_infer"
_PACKAGE_NAME = "_krea2_int8_linear_tests"


def _load_modules():
    names = (
        _PACKAGE_NAME,
        f"{_PACKAGE_NAME}.convrot",
        f"{_PACKAGE_NAME}.int8_linear",
    )
    previous = {name: sys.modules.get(name, _MISSING) for name in names}

    package = types.ModuleType(_PACKAGE_NAME)
    package.__path__ = [str(PACKAGE_ROOT)]
    package.__package__ = _PACKAGE_NAME
    sys.modules[_PACKAGE_NAME] = package

    try:
        loaded = {}
        for short in ("convrot", "int8_linear"):
            module_name = f"{_PACKAGE_NAME}.{short}"
            spec = importlib.util.spec_from_file_location(
                module_name, PACKAGE_ROOT / f"{short}.py"
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load {short}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            loaded[short] = module
        return loaded["convrot"], loaded["int8_linear"]
    finally:
        for name, old in previous.items():
            if old is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


_convrot, _int8 = _load_modules()


class ConvRotTests(unittest.TestCase):
    def test_hadamard_is_orthogonal(self):
        H = _convrot.build_hadamard(16, device="cpu", dtype=torch.float32)
        eye = torch.eye(16)
        self.assertTrue(torch.allclose(H @ H.T, eye, atol=1e-5))

    def test_weight_then_activation_rotation_preserves_matmul(self):
        torch.manual_seed(0)
        group = 16
        out_f, in_f = 8, 32
        W = torch.randn(out_f, in_f)
        x = torch.randn(4, in_f)
        H = _convrot.build_hadamard(group, dtype=W.dtype)
        W_rot = _convrot.rotate_weight(W, H, group)
        x_rot = _convrot.rotate_activation(x, H, group)
        # (x @ H) @ (W @ H^T)^T = x @ H @ H @ W^T, H orthogonal => H@H != I for regular...
        # For orthogonal H: H @ H.T = I. W_rot = W @ H.T, so W_rot.T = H @ W.T
        # x_rot @ W_rot.T = (x @ H) @ (H @ W.T) = x @ (H @ H) @ W.T — need H @ H = I
        # Regular normalized Hadamard is orthogonal: H.T @ H = I, so H @ H.T = I.
        # x_rot @ W_rot.T = x @ H @ (W @ H.T).T = x @ H @ H @ W.T = x @ (H @ H) @ W.T
        # If H is symmetric orthogonal, H@H=I. Regular Hadamard normalized is orthogonal
        # but not always symmetric. W_rot = W @ H.T; x_rot = x @ H
        # x_rot @ W_rot.T = x @ H @ H @ W.T only if W_rot.T = H @ W.T i.e. W_rot = W @ H.T yes
        # = x @ H @ H @ W.T — need H@H = I. For orthogonal: H.T@H=I not H@H=I unless symmetric.
        # Correct: x_rot @ W_rot.T = (x @ H) @ (H @ W.T) = x @ (H @ H) @ W.T
        # H @ H = H @ H. Actually W_rot.T = (W @ H.T).T = H @ W.T. Yes.
        # (x @ H) @ (H @ W.T) = x @ (H @ H) @ W.T. For orthogonal matrices H.T=H^{-1},
        # H @ H is not necessarily I. Wait: (W @ H.T).T = H @ W.T only if H.T.T = H, yes H.
        # H @ W.T where H is the same H. (x@H)@(H@W.T) = x@(H@H)@W.T.
        #
        # Correct identity for orthogonal H: W_rot = W @ H.T, x_rot = x @ H
        # x_rot @ W_rot.T = x @ H @ H @ W.T? W_rot.T = H @ W.T, so x @ H @ H @ W.T
        # We need H @ H = I? No - we need H @ H.T wait:
        # x @ H @ (H @ W.T) = x @ (H @ H) @ W.T — still H@H
        # Actually: W_rot = W @ H.T, W_rot.T = H @ W.T  (since (H.T).T = H)
        # x_rot = x @ H
        # x_rot @ W_rot.T = x @ H @ H @ W.T — H@H not I
        #
        # The correct pair is: W_rot = W @ H.T, x_rot = x @ H
        # Then x_rot @ W_rot.T = x @ H @ (W @ H.T).T = x @ H @ H @ W.T
        #
        # H is orthogonal so H.T @ H = I and H @ H.T = I.
        # (W @ H.T).T = H @ W.T  — yes uses H not H.T
        # x @ H @ H @ W.T — that's H @ H = (H @ H.T) only if H.T=H
        #
        # Let me recalculate:
        # W_rot.T = (W @ H.T).T = H @ W.T   [because (AB).T = B.T A.T, (H.T).T=H]
        # x_rot @ W_rot.T = (x @ H) @ (H @ W.T) = x @ (H @ H) @ W.T
        #
        # H @ H vs H @ H.T: we need H @ H = I for preservation, which requires H = H.T.
        # Looking at H4 base - is it symmetric? Yes H4 is symmetric.
        # Kronecker of symmetric is symmetric. So H is symmetric orthogonal => H@H=I.
        ref = x @ W.T
        got = x_rot @ W_rot.T
        self.assertTrue(torch.allclose(got, ref, atol=1e-4))


class Int8LinearMathTests(unittest.TestCase):
    def test_quantize_axiswise_roundtrip_scale(self):
        x = torch.tensor([[1.0, -2.0, 0.5, 0.0], [3.0, 1.0, -1.0, 2.0]])
        q, scale = _int8.quantize_int8_axiswise(x, dim=-1)
        self.assertEqual(q.dtype, torch.int8)
        recon = q.float() * scale
        # scale is amax/127; reconstruction within ~1/127 of amax per row
        self.assertTrue(torch.allclose(recon, x, atol=0.05))

    def test_int8_forward_matches_dequant_linear(self):
        torch.manual_seed(1)
        in_f, out_f = 32, 16
        weight_fp = torch.randn(out_f, in_f)
        bias = torch.randn(out_f)
        q_w, w_scale = _int8.quantize_int8_axiswise(weight_fp, dim=-1)
        x = torch.randn(5, in_f)

        module = nn.Linear(in_f, out_f, bias=True)
        module.weight = nn.Parameter(q_w, requires_grad=False)
        module.bias = nn.Parameter(bias, requires_grad=False)
        module.register_buffer("weight_scale", w_scale.float())
        module._use_convrot = False

        y = _int8.int8_linear_forward(x, module, torch.float32)
        w_deq = q_w.float() * w_scale.float()
        # Dynamic act quant introduces error vs pure dequant F.linear —
        # compare against dequant path forced (batch small may use dequant)
        y_ref = F.linear(x, w_deq, bias)
        # Force dequant by using small batch path is always for batch<=16
        self.assertTrue(torch.allclose(y, y_ref, atol=1e-4, rtol=1e-4))

    def test_int8_forward_with_convrot_matches_rotated_dequant(self):
        torch.manual_seed(2)
        group = 16
        in_f, out_f = 32, 8
        W = torch.randn(out_f, in_f)
        H = _convrot.build_hadamard(group, dtype=W.dtype)
        W_rot = _convrot.rotate_weight(W, H, group)
        q_w, w_scale = _int8.quantize_int8_axiswise(W_rot, dim=-1)

        module = nn.Linear(in_f, out_f, bias=False)
        module.weight = nn.Parameter(q_w, requires_grad=False)
        module.bias = None
        module.register_buffer("weight_scale", w_scale.float())
        module._use_convrot = True
        module._convrot_groupsize = group

        x = torch.randn(3, in_f)
        y = _int8.int8_linear_forward(x, module, torch.float32)

        x_rot = _convrot.rotate_activation(x, H, group)
        w_deq = q_w.float() * w_scale.float()
        y_ref = F.linear(x_rot, w_deq, None)
        self.assertTrue(torch.allclose(y, y_ref, atol=1e-4, rtol=1e-4))

    def test_is_int8_linear_detects_scale_buffer(self):
        m = nn.Linear(4, 4, bias=False)
        m.weight = nn.Parameter(torch.zeros(4, 4, dtype=torch.int8), requires_grad=False)
        m.register_buffer("weight_scale", torch.ones(4, 1))
        self.assertTrue(_int8.is_int8_linear(m))
        self.assertFalse(_int8.is_int8_linear(nn.Linear(4, 4)))

    def test_parse_comfy_quant_marker(self):
        payload = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
        raw = torch.tensor(list(json.dumps(payload).encode("utf-8")), dtype=torch.uint8)
        conf = _int8.parse_comfy_quant_marker(raw)
        self.assertEqual(conf["format"], "int8_tensorwise")
        self.assertTrue(conf["convrot"])
        self.assertEqual(conf["convrot_groupsize"], 256)

    def test_materialize_int8_state_onto_linear(self):
        # in_features must be divisible by convrot_groupsize for rotation to stay on.
        linear = nn.Linear(16, 4, bias=True)
        weight = torch.randint(-10, 10, (4, 16), dtype=torch.int8)
        scale = torch.ones(4, 1, dtype=torch.float32) * 0.01
        bias = torch.randn(4)
        marker = torch.tensor(
            list(
                json.dumps(
                    {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16}
                ).encode("utf-8")
            ),
            dtype=torch.uint8,
        )
        _int8.materialize_int8_linear(
            linear,
            weight=weight,
            weight_scale=scale,
            bias=bias,
            comfy_quant=marker,
        )
        self.assertEqual(linear.weight.dtype, torch.int8)
        self.assertTrue(torch.equal(linear.weight, weight))
        self.assertTrue(torch.allclose(linear.weight_scale, scale))
        self.assertTrue(linear._use_convrot)
        self.assertEqual(linear._convrot_groupsize, 16)


class PartitionInt8StateDictTests(unittest.TestCase):
    def test_splits_quant_side_tensors(self):
        state = {
            "foo.weight": torch.zeros(2, 2, dtype=torch.int8),
            "foo.weight_scale": torch.ones(2, 1),
            "foo.comfy_quant": torch.tensor([1, 2], dtype=torch.uint8),
            "foo.bias": torch.zeros(2),
            "bar.weight": torch.randn(2, 2),
            "bar.input_scale": torch.tensor(1.0),
        }
        weights, side = _int8.partition_int8_state_dict(state)
        self.assertIn("foo.weight", weights)
        self.assertIn("foo.bias", weights)
        self.assertIn("bar.weight", weights)
        self.assertNotIn("foo.weight_scale", weights)
        self.assertIn("foo.weight_scale", side)
        self.assertIn("foo.comfy_quant", side)
        self.assertIn("bar.input_scale", side)

    def test_detects_int8_tensorwise_presence(self):
        state = {
            "a.weight": torch.zeros(2, 2, dtype=torch.int8),
            "a.weight_scale": torch.ones(2, 1),
        }
        self.assertTrue(_int8.state_dict_has_int8_tensorwise(state))
        self.assertFalse(
            _int8.state_dict_has_int8_tensorwise(
                {"a.weight": torch.randn(2, 2)}
            )
        )


if __name__ == "__main__":
    unittest.main()
