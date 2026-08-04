# workers/minimax_h3/tests/test_comfy_dit_load.py
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

from h3_infer.comfy_dit_load import load_comfy_int8_dit
from h3_infer.minimax_h3_convert import (
    MINIMAX_H3_TEST_TRANSFORMER_CONFIG as CFG,
    SourceLayout,
)
from h3_infer.meta_init import materialize_nonpersistent_buffers


def _write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    save_file(tensors, str(path))


class TinyDiffusersLike(nn.Module):
    """Minimal module with post-convert Linear names for one block QKV."""

    def __init__(self, hidden: int, inner: int):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.ModuleDict(
                            {
                                "to_q": nn.Linear(hidden, inner, bias=False),
                                "to_k": nn.Linear(hidden, inner, bias=False),
                                "to_v": nn.Linear(hidden, inner, bias=False),
                            }
                        )
                    }
                )
            ]
        )

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        # Flatten ModuleDict paths to dotted names used by load_state_dict
        return super().state_dict(*args, **kwargs)


class FlatTiny(nn.Module):
    def __init__(self, hidden: int, inner: int):
        super().__init__()
        # Use nested modules with real attribute path
        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(hidden, inner, bias=False)
                self.to_k = nn.Linear(hidden, inner, bias=False)
                self.to_v = nn.Linear(hidden, inner, bias=False)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attn()

        self.transformer_blocks = nn.ModuleList([Block()])


class TestComfyDitLoad(unittest.TestCase):
    def test_g0_nogo_raises(self):
        heads, hd = CFG["num_attention_heads"], CFG["attention_head_dim"]
        hidden = CFG["hidden_size"]
        rows = 3 * heads * hd
        w = torch.randint(-2, 2, (rows, hidden), dtype=torch.int8)
        # Curve checkpoint: narrow adaln + table
        state = {
            "adaln_t_table": torch.zeros(8, 8),
            "blocks.0.adaln_proj.linear.weight": torch.randint(
                -2, 2, (6 * 3 * hidden, 8), dtype=torch.int8
            ),
            "blocks.0.attn.qkv_proj.weight": w,
            "blocks.0.attn.qkv_proj.weight_scale": torch.ones(rows),
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "curve.safetensors"
            _write_safetensors(path, state)
            m = FlatTiny(hidden, heads * hd)
            m.requires_grad_(False)
            with self.assertRaises(RuntimeError) as ctx:
                load_comfy_int8_dit(
                    m,
                    path,
                    config=CFG,
                    source_layout=SourceLayout.COMFY_QKV_CONTIGUOUS,
                    min_source_coverage=0.0,
                )
            self.assertIn("G0 FAIL", str(ctx.exception))

    def test_assign_load_int8_qkv(self):
        heads, hd = CFG["num_attention_heads"], CFG["attention_head_dim"]
        hidden = CFG["hidden_size"]
        rows = 3 * heads * hd
        w = torch.randint(-3, 3, (rows, hidden), dtype=torch.int8)
        scale = torch.ones(rows)
        # G0 needs full adaln evidence
        state = {
            "time_embedder.proj_out.weight": torch.randn(
                CFG["time_embed_dim"], CFG["time_embed_hidden_dim"]
            ),
            "blocks.0.adaln_proj.linear.weight": torch.randint(
                -2, 2, (6 * 3 * hidden, CFG["time_embed_dim"]), dtype=torch.int8
            ),
            "blocks.0.adaln_proj.linear.weight_scale": torch.ones(6 * 3 * hidden),
            "blocks.0.attn.qkv_proj.weight": w,
            "blocks.0.attn.qkv_proj.weight_scale": scale,
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ok.safetensors"
            _write_safetensors(path, state)
            m = FlatTiny(hidden, heads * hd)
            m.requires_grad_(False)
            # G3: required keys from full plan won't all be present on tiny module.
            # Patch load path by using a module that only has qkv targets and
            # lower min_source_coverage; G3 checks module_keys ∩ required.
            info = load_comfy_int8_dit(
                m,
                path,
                config=CFG,
                source_layout=SourceLayout.COMFY_QKV_CONTIGUOUS,
                min_source_coverage=0.3,
                compute_dtype=torch.float32,
                device="cpu",
            )
            self.assertEqual(m.transformer_blocks[0].attn.to_q.weight.dtype, torch.int8)
            self.assertGreaterEqual(info["int8_layers"], 1)
            self.assertGreaterEqual(info["patched_linears"], 1)
            # Forward through patched linear
            x = torch.randn(2, hidden)
            y = m.transformer_blocks[0].attn.to_q(x)
            self.assertTrue(torch.isfinite(y).all())

    def test_materialize_idempotent(self):
        class Stub(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("inv_freq", torch.empty(0), persistent=False)

        class Rope(nn.Module):
            def __init__(self):
                super().__init__()

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.rope = Rope()
                # start with meta-like empty — simulate missing inv_freq
                self.config = type(
                    "C",
                    (),
                    {"rope_theta": 10000.0, "rope_freq_dim": 4},
                )()

        m = M()
        materialize_nonpersistent_buffers(m, "cpu")
        a = m.rope.inv_freq.clone()
        materialize_nonpersistent_buffers(m, "cpu")
        b = m.rope.inv_freq
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(b.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
