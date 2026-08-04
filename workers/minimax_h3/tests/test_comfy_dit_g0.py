# workers/minimax_h3/tests/test_comfy_dit_g0.py
from __future__ import annotations

import unittest

from h3_infer.comfy_dit_g0 import classify_dit_checkpoint, G0Result


class TestG0(unittest.TestCase):
    def test_curve_table_is_nogo(self):
        header = {
            "adaln_t_table": {"dtype": "F32", "shape": [1025, 8]},
            "blocks.0.adaln_proj.linear.weight": {"dtype": "I8", "shape": [96768, 8]},
        }
        r = classify_dit_checkpoint(header)
        self.assertEqual(r.verdict, "NO_GO_CURVE_ADALN")
        self.assertFalse(r.compatible_with_stock_diffusers)

    def test_full_time_embed_ok(self):
        # time_embed_dim=2688; adaln out = 6 * 3 * hidden = 6*3*5376 = 96768, in = 2688
        header = {
            "time_embedder.proj_in.weight": {"dtype": "F32", "shape": [5376, 256]},
            "time_embedder.proj_out.weight": {"dtype": "F32", "shape": [2688, 5376]},
            "blocks.0.adaln_proj.linear.weight": {"dtype": "I8", "shape": [96768, 2688]},
            "blocks.0.attn.qkv_proj.weight": {"dtype": "I8", "shape": [21504, 5376]},
        }
        r = classify_dit_checkpoint(header)
        self.assertEqual(r.verdict, "OK_FULL_ADALN")
        self.assertTrue(r.compatible_with_stock_diffusers)

    def test_narrow_adaln_without_table_still_nogo(self):
        header = {
            "blocks.0.adaln_proj.linear.weight": {"dtype": "I8", "shape": [96768, 8]},
        }
        r = classify_dit_checkpoint(header)
        self.assertFalse(r.compatible_with_stock_diffusers)
        self.assertEqual(r.verdict, "NO_GO_NARROW_ADALN")

    def test_inconclusive_without_signals(self):
        header = {
            "blocks.0.attn.qkv_proj.weight": {"dtype": "I8", "shape": [21504, 5376]},
        }
        r = classify_dit_checkpoint(header)
        self.assertEqual(r.verdict, "INCONCLUSIVE")
        self.assertFalse(r.compatible_with_stock_diffusers)
        self.assertIsInstance(r, G0Result)


if __name__ == "__main__":
    unittest.main()
