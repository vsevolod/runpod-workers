import math
import sys
import types
import unittest
from pathlib import Path

import torch

_PACKAGE_ROOT = Path(__file__).parents[1] / "krea2_infer"


def _ensure_krea2_infer_namespace():
    """Register krea2_infer without executing package __init__ (avoids heavy deps)."""
    name = "krea2_infer"
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__path__", None):
        # If full package already loaded, reuse it.
        if hasattr(existing, "edit_sampling") or name + ".edit_sampling" in sys.modules:
            return
        # Heavy package __init__ may have partially failed — replace only if needed.
        if not getattr(existing, "__file__", None) or existing.__file__ is None:
            return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(_PACKAGE_ROOT)]
    sys.modules[name] = pkg


_ensure_krea2_infer_namespace()

from krea2_infer.edit_sampling import (  # noqa: E402
    build_edit_position_ids,
    build_ref_boost_bias,
)


class PositionIdsTests(unittest.TestCase):
    def test_single_ref_full_grid_frames(self):
        # text=4, src grid 2x2, tgt 2x2, pos_mode=anchor
        pos = build_edit_position_ids(
            batch=1,
            text_len=4,
            src_grids=[(2, 2)],
            tgt_grid=(2, 2),
            pos_mode="anchor",
            device=torch.device("cpu"),
        )
        # shape (B, T+S+G, 3) = (1, 4+4+4, 3)
        self.assertEqual(tuple(pos.shape), (1, 12, 3))
        self.assertTrue(torch.equal(pos[0, :4], torch.zeros(4, 3)))
        self.assertTrue(torch.all(pos[0, 4:8, 0] == 1))  # source frame
        self.assertTrue(torch.all(pos[0, 8:12, 0] == 0))  # target frame

    def test_fit_offset_centers_smaller_src(self):
        # src 2x2 inside tgt 4x4 → off_h=off_w=1
        pos = build_edit_position_ids(
            batch=1,
            text_len=1,
            src_grids=[(2, 2)],
            tgt_grid=(4, 4),
            pos_mode="stride1",
            device=torch.device("cpu"),
        )
        src = pos[0, 1:5]  # 4 src tokens
        # h coords in {1,2}, w in {1,2}
        self.assertEqual(set(src[:, 1].tolist()), {1.0, 2.0})
        self.assertEqual(set(src[:, 2].tolist()), {1.0, 2.0})


class RefBoostBiasTests(unittest.TestCase):
    def test_none_when_boost_one(self):
        self.assertIsNone(
            build_ref_boost_bias(
                [1.0],
                text_len=3,
                src_token_lens=[4],
                tgt_len=4,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        )

    def test_log_boost_on_target_to_src(self):
        bias = build_ref_boost_bias(
            [4.0],
            text_len=2,
            src_token_lens=[3],
            tgt_len=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        # L=2+3+4=9
        self.assertEqual(tuple(bias.shape), (1, 1, 9, 9))
        expected = math.log(4.0)
        # target rows 5:9, src cols 2:5
        self.assertTrue(
            torch.allclose(bias[0, 0, 5:9, 2:5], torch.full((4, 3), expected))
        )
        self.assertTrue(torch.all(bias[0, 0, :5, :] == 0))


class VelocitySliceTests(unittest.TestCase):
    def test_velocity_slice_keeps_only_target(self):
        # src_len=4, tgt_len=6, model returns (1, 10, D)
        full = torch.arange(10).view(1, 10, 1).float()
        tgt = full[:, -6:, :]
        self.assertEqual(tgt.shape[1], 6)
        self.assertEqual(tgt[0, 0, 0].item(), 4.0)


if __name__ == "__main__":
    unittest.main()
