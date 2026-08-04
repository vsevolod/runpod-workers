# workers/minimax_h3/tests/test_minimax_h3_convert.py
from __future__ import annotations

import unittest

import torch

from h3_infer.minimax_h3_convert import (
    MINIMAX_H3_TEST_TRANSFORMER_CONFIG as CFG,
    SourceLayout,
    convert_transformer_key,
    convert_transformer_key_with_sides,
    g1_int8_source_coverage,
    reorder_interleaved_qkv,
    split_fused_qkv,
)


def _interleave_qkv(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, hd: int
) -> torch.Tensor:
    """Inverse of reorder_interleaved_qkv: pack per-head [q,k,v] rows."""
    # q,k,v each [heads*hd, hidden]
    qh = q.reshape(heads, hd, *q.shape[1:])
    kh = k.reshape(heads, hd, *k.shape[1:])
    vh = v.reshape(heads, hd, *v.shape[1:])
    # per head: [q_hd; k_hd; v_hd] along dim1
    per_head = torch.cat([qh, kh, vh], dim=1)  # [heads, 3*hd, ...]
    return per_head.reshape(heads * 3 * hd, *q.shape[1:])


class TestConvert(unittest.TestCase):
    def test_comfy_layout_splits_without_reorder(self):
        """Input is already [q_all;k_all;v_all]; layout=COMFY must match plain split."""
        heads, hd = CFG["num_attention_heads"], CFG["attention_head_dim"]
        hidden = CFG["hidden_size"]
        q = torch.arange(heads * hd * hidden, dtype=torch.float32).reshape(
            heads * hd, hidden
        )
        k = q + 1000
        v = q + 2000
        w = torch.cat([q, k, v], dim=0)
        outs = convert_transformer_key_with_sides(
            {"blocks.0.attn.qkv_proj.weight": w},
            CFG,
            source_layout=SourceLayout.COMFY_QKV_CONTIGUOUS,
        )
        self.assertTrue(
            torch.equal(outs["transformer_blocks.0.attn.to_q.weight"], q)
        )
        self.assertTrue(
            torch.equal(outs["transformer_blocks.0.attn.to_k.weight"], k)
        )
        self.assertTrue(
            torch.equal(outs["transformer_blocks.0.attn.to_v.weight"], v)
        )

    def test_official_layout_reorders_then_splits(self):
        heads, hd = CFG["num_attention_heads"], CFG["attention_head_dim"]
        hidden = CFG["hidden_size"]
        inner = heads * hd
        q = torch.randn(inner, hidden)
        k = torch.randn(inner, hidden)
        v = torch.randn(inner, hidden)
        raw = _interleave_qkv(q, k, v, heads, hd)
        outs = convert_transformer_key_with_sides(
            {"blocks.0.attn.qkv_proj.weight": raw},
            CFG,
            source_layout=SourceLayout.OFFICIAL_RAW_INTERLEAVED,
        )
        reordered = reorder_interleaved_qkv(raw, heads, hd)
        tq, tk, tv = split_fused_qkv(reordered, heads, hd)
        self.assertTrue(
            torch.equal(outs["transformer_blocks.0.attn.to_q.weight"], tq)
        )
        self.assertTrue(
            torch.equal(outs["transformer_blocks.0.attn.to_k.weight"], tk)
        )
        self.assertTrue(
            torch.equal(outs["transformer_blocks.0.attn.to_v.weight"], tv)
        )

    def test_mlp_fc1_half_swap(self):
        ffn, hidden = CFG["ffn_dim"], CFG["hidden_size"]
        gate = torch.ones(ffn, hidden)
        value = torch.full((ffn, hidden), 2.0)
        w = torch.cat([gate, value], dim=0)
        outs = convert_transformer_key("blocks.0.mlp.fc1.weight", w, CFG)
        self.assertEqual(outs[0][0], "transformer_blocks.0.ff.net.0.proj.weight")
        out = outs[0][1]
        self.assertTrue(torch.equal(out[:ffn], value))
        self.assertTrue(torch.equal(out[ffn:], gate))

    def test_int8_qkv_scale_splits_comfy_layout(self):
        heads, hd = CFG["num_attention_heads"], CFG["attention_head_dim"]
        hidden = CFG["hidden_size"]
        rows = 3 * heads * hd
        w = torch.randint(-8, 8, (rows, hidden), dtype=torch.int8)
        scale = torch.arange(rows, dtype=torch.float32)
        marker = torch.tensor(
            list(b'{"convrot":true,"convrot_groupsize":4}'), dtype=torch.uint8
        )
        state = {
            "blocks.0.attn.qkv_proj.weight": w,
            "blocks.0.attn.qkv_proj.weight_scale": scale,
            "blocks.0.attn.qkv_proj.comfy_quant": marker,
        }
        converted = convert_transformer_key_with_sides(
            state, CFG, source_layout=SourceLayout.COMFY_QKV_CONTIGUOUS
        )
        third = heads * hd
        self.assertTrue(
            torch.equal(
                converted["transformer_blocks.0.attn.to_q.weight_scale"],
                scale[:third],
            )
        )
        self.assertIn("transformer_blocks.0.attn.to_q.comfy_quant", converted)

    def test_int8_mlp_fc1_scale_half_swap(self):
        ffn, hidden = CFG["ffn_dim"], CFG["hidden_size"]
        w = torch.randint(-4, 4, (2 * ffn, hidden), dtype=torch.int8)
        scale = torch.cat([torch.ones(ffn), torch.full((ffn,), 3.0)])
        converted = convert_transformer_key_with_sides(
            {
                "blocks.0.mlp.fc1.weight": w,
                "blocks.0.mlp.fc1.weight_scale": scale,
            },
            CFG,
            source_layout=SourceLayout.COMFY_QKV_CONTIGUOUS,
        )
        key = "transformer_blocks.0.ff.net.0.proj.weight_scale"
        self.assertTrue(torch.equal(converted[key][:ffn], torch.full((ffn,), 3.0)))
        self.assertTrue(torch.equal(converted[key][ffn:], torch.ones(ffn)))

    def test_g1_unknown_int8_weight_lowers_coverage(self):
        """Denominator = all int8 .weight in state, not planned∩present."""
        heads, hd = CFG["num_attention_heads"], CFG["attention_head_dim"]
        hidden = CFG["hidden_size"]
        rows = 3 * heads * hd
        good = torch.randint(-2, 2, (rows, hidden), dtype=torch.int8)
        orphan = torch.randint(-2, 2, (3, hidden), dtype=torch.int8)
        state = {
            "blocks.0.attn.qkv_proj.weight": good,
            "blocks.0.attn.qkv_proj.weight_scale": torch.ones(rows),
            "totally_unknown.int8.weight": orphan,
        }
        rep = g1_int8_source_coverage(
            state, CFG, source_layout=SourceLayout.COMFY_QKV_CONTIGUOUS
        )
        self.assertEqual(rep["int8_weight_total"], 2)
        self.assertEqual(rep["int8_weight_mapped_ok"], 1)
        self.assertAlmostEqual(rep["source_coverage_ratio"], 0.5)
        self.assertTrue(any("totally_unknown" in u for u in rep["unmapped_int8_weights"]))


if __name__ == "__main__":
    unittest.main()
