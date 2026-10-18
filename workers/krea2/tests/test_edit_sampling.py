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
    resolve_edit_canvas_size,
    sample_edit,
)


class PositionIdsTests(unittest.TestCase):
    def test_fit_odd_gap_centers_at_half_token_without_shifting_reference(self):
        pos = build_edit_position_ids(1, 1, [(2, 2)], (5, 5),
                                      pos_mode="stride1", device="cpu")
        self.assertEqual(pos[0, 1:5, 1:].tolist(),
                         [[1.5, 1.5], [1.5, 2.5], [2.5, 1.5], [2.5, 2.5]])

    def test_scene_aligned_person_beside_keeps_target_and_scene_positions(self):
        pos = build_edit_position_ids(1, 1, [(2, 2), (1, 2)], (3, 3),
                                      pos_mode="stride1", device="cpu",
                                      ref_position="beside", ref_position_a="aligned")
        self.assertEqual(pos[0, 1:5].tolist(),
                         [[1., .5, .5], [1., .5, 1.5], [1., 1.5, .5], [1., 1.5, 1.5]])
        self.assertEqual(pos[0, 5:7].tolist(), [[2., 1., 3.], [2., 1., 4.]])
        self.assertEqual(pos[0, 7].tolist(), [0., 0., 0.])
        self.assertEqual(pos[0, -1].tolist(), [0., 2., 2.])

    def test_two_beside_refs_do_not_overlap_each_other(self):
        pos = build_edit_position_ids(1, 0, [(1, 2), (1, 3)], (2, 4),
                                      pos_mode="stride1", device="cpu",
                                      ref_position="beside", ref_position_a="beside")
        self.assertEqual(pos[0, :5, 2].tolist(), [4., 5., 6., 7., 8.])

    def test_bag_of_features_collapses_layout_but_preserves_reference_frame(self):
        pos = build_edit_position_ids(1, 0, [(2, 2)], (3, 5),
                                      pos_mode="stride1", device="cpu",
                                      ref_position="bag_of_features")
        self.assertEqual(pos[0, :4].tolist(), [[1., 1., 2.]] * 4)

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
    def test_scene_and_person_have_independent_attention_boosts(self):
        bias = build_ref_boost_bias([1., 3.], 1, [2, 3], 4,
                                    device="cpu", dtype=torch.float32)
        self.assertTrue(torch.all(bias[0, 0, 6:, 1:3] == 0))
        torch.testing.assert_close(bias[0, 0, 6:, 3:6], torch.full((4, 3), math.log(3.)))

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

    def test_two_sources_boost_spans_both_blocks(self):
        bias = build_ref_boost_bias(
            [4.0, 4.0],
            text_len=1,
            src_token_lens=[2, 3],
            tgt_len=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        # L = 1+2+3+4 = 10
        self.assertEqual(tuple(bias.shape), (1, 1, 10, 10))
        expected = math.log(4.0)
        # target rows 6:10, src0 cols 1:3, src1 cols 3:6
        self.assertTrue(
            torch.allclose(bias[0, 0, 6:10, 1:3], torch.full((4, 2), expected))
        )
        self.assertTrue(
            torch.allclose(bias[0, 0, 6:10, 3:6], torch.full((4, 3), expected))
        )


class SamplerV2ControlsTests(unittest.TestCase):
    def test_cfg_uses_identical_placement_and_independent_boosts_in_both_branches(self):
        from PIL import Image
        class Model(torch.nn.Module):
            config = types.SimpleNamespace(patch=1)
            def __init__(self):
                super().__init__()
                self.calls = []
            def forward(self, *, img, pos, attn_bias, **kwargs):
                self.calls.append((pos.clone(), attn_bias.clone()))
                return torch.zeros_like(img)
        class Encoder(torch.nn.Module):
            def grounded_encode(self, text, images, **kwargs):
                return torch.zeros(1, 1, 1, 1), torch.ones(1, 1, dtype=torch.bool)
        class Autoencoder:
            compression = 1
            channels = 3
            def encode(self, image): return image
            def decode(self, latent): return latent
        model = Model()
        images = sample_edit(model, Autoencoder(), Encoder(), "place person", [Image.new("RGB", (16, 16))] * 2,
                             device="cpu", dtype=torch.float32, width=16, height=16, steps=1, guidance=2.,
                             ref_position="beside", ref_position_a="aligned", ref_boost=3., ref_boost_a=1.)
        self.assertEqual(len(images), 1)
        self.assertEqual(len(model.calls), 2)
        for pos, bias in model.calls:
            self.assertEqual(pos[0, 1].tolist(), [1., 0., 0.])
            self.assertEqual(pos[0, 257].tolist(), [2., 0., 16.])
            self.assertEqual(pos[0, 513].tolist(), [0., 0., 0.])
            self.assertTrue(torch.all(bias[0, 0, 513:, 1:257] == 0))
            torch.testing.assert_close(bias[0, 0, 513:, 257:513], torch.full((256, 256), math.log(3.)))


class MultiRefPositionIdsTests(unittest.TestCase):
    def test_two_sources_frames_and_total_len(self):
        # text=2, src0 2x2, src1 1x2, tgt 3x3
        pos = build_edit_position_ids(
            batch=1,
            text_len=2,
            src_grids=[(2, 2), (1, 2)],
            tgt_grid=(3, 3),
            pos_mode="anchor",
            device=torch.device("cpu"),
        )
        # 2 + 4 + 2 + 9 = 17
        self.assertEqual(tuple(pos.shape), (1, 17, 3))
        self.assertTrue(torch.all(pos[0, 2:6, 0] == 1.0))  # frame 1
        self.assertTrue(torch.all(pos[0, 6:8, 0] == 2.0))  # frame 2
        self.assertTrue(torch.all(pos[0, 8:17, 0] == 0.0))  # target


class ResolveCanvasSizeTests(unittest.TestCase):
    def test_xor_raises(self):
        from PIL import Image

        src = Image.new("RGB", (64, 64))
        with self.assertRaisesRegex(ValueError, "both"):
            resolve_edit_canvas_size([src], 512, None)
        with self.assertRaisesRegex(ValueError, "both"):
            resolve_edit_canvas_size([src], None, 512)

    def test_both_none_uses_first_source(self):
        from PIL import Image

        a = Image.new("RGB", (320, 160))
        b = Image.new("RGB", (999, 999))
        w, h = resolve_edit_canvas_size(
            [a, b], None, None, max_megapixels=1.5, align=16
        )
        self.assertEqual(w % 16, 0)
        self.assertEqual(h % 16, 0)
        self.assertAlmostEqual(w / h, 320 / 160, delta=0.05)

    def test_both_int_roundup(self):
        from PIL import Image

        src = Image.new("RGB", (10, 10))
        w, h = resolve_edit_canvas_size([src], 1000, 500, align=16)
        self.assertEqual(w % 16, 0)
        self.assertEqual(h % 16, 0)
        self.assertGreaterEqual(w, 1000)
        self.assertGreaterEqual(h, 500)


class VelocitySliceTests(unittest.TestCase):
    def test_velocity_slice_keeps_only_target(self):
        # src_len=4, tgt_len=6, model returns (1, 10, D)
        full = torch.arange(10).view(1, 10, 1).float()
        tgt = full[:, -6:, :]
        self.assertEqual(tgt.shape[1], 6)
        self.assertEqual(tgt[0, 0, 0].item(), 4.0)


if __name__ == "__main__":
    unittest.main()
