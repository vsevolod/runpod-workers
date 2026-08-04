# workers/minimax_h3/tests/test_int8_assign_and_patch.py
from __future__ import annotations

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from h3_infer.int8_linear import (
    apply_int8_side_tensors,
    is_int8_linear,
    partition_int8_state_dict,
)
from h3_infer.int8_linear_patch import patch_module_int8_linears


class TestAssignAndPatch(unittest.TestCase):
    def test_assign_true_keeps_int8_dtype(self):
        class Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(8, 4, bias=False)

        wrap = Wrap()
        wrap.requires_grad_(False)
        w = torch.randint(-7, 7, (4, 8), dtype=torch.int8)
        scale = torch.ones(4)
        incompatible = wrap.load_state_dict(
            {"proj.weight": w}, strict=False, assign=True
        )
        self.assertEqual(len(incompatible.missing_keys), 0)
        self.assertEqual(wrap.proj.weight.dtype, torch.int8)
        n = apply_int8_side_tensors(
            wrap, {"proj.weight_scale": scale}, {"proj.weight": w}
        )
        self.assertEqual(n, 1)
        self.assertTrue(is_int8_linear(wrap.proj))
        # partition helper sanity
        weights, side = partition_int8_state_dict(
            {"proj.weight": w, "proj.weight_scale": scale}
        )
        self.assertIn("proj.weight", weights)
        self.assertIn("proj.weight_scale", side)

    def test_patch_matches_dequant_reference(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(8, 4, bias=False)

        m = M()
        m.requires_grad_(False)
        w = torch.randint(-5, 5, (4, 8), dtype=torch.int8)
        scale = torch.linspace(0.01, 0.04, 4)
        m.load_state_dict({"proj.weight": w}, strict=True, assign=True)
        apply_int8_side_tensors(
            m,
            {"proj.weight_scale": scale},
            {"proj.weight": w},
        )
        patch_module_int8_linears(m, compute_dtype=torch.float32)
        x = torch.randn(2, 8)
        y = m.proj(x)
        w_f = w.float() * scale.float().unsqueeze(1)
        y_ref = F.linear(x.float(), w_f)
        self.assertTrue(torch.allclose(y.float(), y_ref, rtol=1e-4, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
