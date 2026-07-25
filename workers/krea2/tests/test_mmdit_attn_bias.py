import importlib.util
import sys
import unittest
from pathlib import Path

import torch

_MISSING = object()
_PACKAGE_ROOT = Path(__file__).parents[1] / "krea2_infer"
_MODULE_NAME = "_krea2_mmdit_attn_helpers_tests"


def _load_mmdit_helpers():
    """Load pure mask helpers from mmdit without full model construction."""
    # mmdit imports torch only — fine for unit tests. Avoid package __init__.
    spec = importlib.util.spec_from_file_location(
        _MODULE_NAME,
        _PACKAGE_ROOT / "mmdit.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load mmdit module")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(_MODULE_NAME, _MISSING)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is _MISSING:
            sys.modules.pop(_MODULE_NAME, None)
        else:
            sys.modules[_MODULE_NAME] = previous
    return module


_m = _load_mmdit_helpers()
_mask = _m._mask
pad_seq_and_bias = _m.pad_seq_and_bias
select_attn_mask = _m.select_attn_mask
_bool_mask_to_additive = _m._bool_mask_to_additive


class PadSeqAndBiasTests(unittest.TestCase):
    def test_pad_attn_bias_to_multiple_of_256(self):
        L = 300
        padlen_expected = 212
        pos = torch.zeros(1, L, 3)
        bias = torch.zeros(1, 1, L, L)
        mask = torch.ones(1, L, dtype=torch.bool)
        padlen, mask2, pos2, bias2 = pad_seq_and_bias(L, mask, pos, bias, multiple=256)
        self.assertEqual(padlen, padlen_expected)
        self.assertEqual(tuple(pos2.shape), (1, 512, 3))
        self.assertEqual(tuple(mask2.shape), (1, 512))
        self.assertEqual(tuple(bias2.shape), (1, 1, 512, 512))
        self.assertFalse(bool(mask2[0, L:].all()))  # pad keys invalid
        self.assertTrue(torch.all(bias2[0, 0, L:, :] == 0))
        self.assertTrue(torch.all(bias2[0, 0, :, L:] == 0))


class SelectAttnMaskTests(unittest.TestCase):
    def test_legacy_bool_path_when_no_bias(self):
        mask = torch.tensor([[True, True, False]])
        out = select_attn_mask(mask, None, dtype=torch.float32)
        expected = _mask(mask)
        self.assertTrue(torch.equal(out, expected))

    def test_none_mask_none_bias(self):
        self.assertIsNone(select_attn_mask(None, None, dtype=torch.float32))

    def test_bias_only_when_mask_none(self):
        bias = torch.zeros(1, 1, 4, 4)
        bias[0, 0, 2:, 1:2] = 1.5
        out = select_attn_mask(None, bias, dtype=torch.float32)
        self.assertTrue(torch.allclose(out, bias))

    def test_float_combine_pad_and_bias(self):
        mask = torch.tensor([[True, True, False, True]])
        bias = torch.zeros(1, 1, 4, 4)
        bias[0, 0, 3, 0] = 2.0
        out = select_attn_mask(mask, bias, dtype=torch.float32)
        # invalid key col 2 is -inf for all queries
        self.assertTrue(torch.isneginf(out[0, 0, :, 2]).all())
        self.assertEqual(float(out[0, 0, 3, 0]), 2.0)


if __name__ == "__main__":
    unittest.main()
