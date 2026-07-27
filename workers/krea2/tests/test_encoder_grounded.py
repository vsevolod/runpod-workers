import importlib.util
import sys
import types
import unittest
from pathlib import Path

from PIL import Image

_PACKAGE_ROOT = Path(__file__).parents[1] / "krea2_infer"


def _ensure_krea2_infer_namespace():
    name = "krea2_infer"
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__path__", None):
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(_PACKAGE_ROOT)]
    sys.modules[name] = pkg


_ensure_krea2_infer_namespace()

from krea2_infer.grounded import (  # noqa: E402
    grounded_encode_impl,
    grounded_template,
    prepare_grounded_images,
    resize_for_grounding,
)


class GroundedHelperTests(unittest.TestCase):
    def test_grounded_template_inserts_instruction(self):
        t = grounded_template("recolor the car")
        self.assertIn("recolor the car", t)
        self.assertEqual(t.count("<|vision_start|><|image_pad|><|vision_end|>"), 1)

    def test_grounded_template_two_images(self):
        t = grounded_template("combine", num_images=2)
        self.assertEqual(t.count("<|vision_start|><|image_pad|><|vision_end|>"), 2)
        first = t.index("<|image_pad|>")
        second = t.index("<|image_pad|>", first + 1)
        inst = t.index("combine")
        self.assertLess(second, inst)

    def test_grounded_template_rejects_bad_count(self):
        with self.assertRaises(ValueError):
            grounded_template("x", num_images=0)
        with self.assertRaises(ValueError):
            grounded_template("x", num_images=3)

    def test_resize_for_grounding_caps_long_side(self):
        img = Image.new("RGB", (2000, 1000))
        out = resize_for_grounding(img, 768)
        self.assertEqual(max(out.size), 768)

    def test_resize_for_grounding_noop_when_small(self):
        img = Image.new("RGB", (100, 80))
        out = resize_for_grounding(img, 768)
        self.assertEqual(out.size, (100, 80))

    def test_prepare_grounded_images_order_and_count(self):
        a = Image.new("RGB", (2000, 1000), (1, 2, 3))
        b = Image.new("RGB", (500, 1500), (4, 5, 6))
        out = prepare_grounded_images([a, b], 768)
        self.assertEqual(len(out), 2)
        self.assertEqual(max(out[0].size), 768)
        self.assertEqual(max(out[1].size), 768)
        self.assertGreater(out[0].size[0], out[0].size[1])
        self.assertLess(out[1].size[0], out[1].size[1])

    def test_prepare_grounded_images_rejects_0_and_3(self):
        img = Image.new("RGB", (32, 32))
        with self.assertRaises(ValueError):
            prepare_grounded_images([], 768)
        with self.assertRaises(ValueError):
            prepare_grounded_images([img, img, img], 768)


class GroundedEncodeFakeProcessorTests(unittest.TestCase):
    """CPU unit tests for grounded_encode_impl (no real Qwen / transformers)."""

    def test_grounded_encode_two_images_preserves_order_and_pad_count(self):
        import torch

        calls = {}

        def fake_processor(*, text, images, padding, return_tensors):
            calls["text"] = text
            calls["images"] = list(images)
            seq_len = 10
            ids = torch.zeros(1, seq_len, dtype=torch.long)
            mask = torch.ones(1, seq_len, dtype=torch.long)
            pv = torch.zeros(len(images), 3, 8, 8)
            return {
                "input_ids": ids,
                "attention_mask": mask,
                "pixel_values": pv,
                "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
            }

        class FakeQwen(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._p = torch.nn.Parameter(torch.zeros(1))

            def forward(self, **kwargs):
                seq = kwargs["input_ids"].shape[1]
                hs0 = torch.randn(1, seq, 4)
                return type("Out", (), {"hidden_states": (hs0,)})()

        img0 = Image.new("RGB", (2000, 1000), (10, 20, 30))
        img1 = Image.new("RGB", (100, 300), (40, 50, 60))
        hiddens, mask = grounded_encode_impl(
            "put person by tractor",
            [img0, img1],
            grounding_px=768,
            mm_processor=fake_processor,
            qwen=FakeQwen(),
            select_layers=(0,),
            prefix_idx=2,
        )
        self.assertEqual(len(calls["images"]), 2)
        self.assertEqual(calls["text"][0].count("<|image_pad|>"), 2)
        self.assertIn("put person by tractor", calls["text"][0])
        # img0 was oversized → long side capped; img1 100x300 stays under cap
        self.assertEqual(max(calls["images"][0].size), 768)
        self.assertEqual(calls["images"][1].size, (100, 300))
        self.assertGreater(calls["images"][0].size[0], calls["images"][0].size[1])
        self.assertLess(calls["images"][1].size[0], calls["images"][1].size[1])
        self.assertEqual(tuple(hiddens.shape), (1, 8, 1, 4))
        self.assertEqual(tuple(mask.shape), (1, 8))
        self.assertGreater(hiddens.shape[1], 0)

    def test_grounded_encode_rejects_0_and_3(self):
        import torch

        def fake_processor(**kwargs):
            raise AssertionError("processor should not be called")

        class FakeQwen(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._p = torch.nn.Parameter(torch.zeros(1))

        with self.assertRaises(ValueError):
            grounded_encode_impl(
                "x",
                [],
                grounding_px=768,
                mm_processor=fake_processor,
                qwen=FakeQwen(),
                select_layers=(0,),
                prefix_idx=2,
            )
        img = Image.new("RGB", (32, 32))
        with self.assertRaises(ValueError):
            grounded_encode_impl(
                "x",
                [img, img, img],
                grounding_px=768,
                mm_processor=fake_processor,
                qwen=FakeQwen(),
                select_layers=(0,),
                prefix_idx=2,
            )

    def test_grounded_encode_one_image_regression(self):
        import torch

        calls = {}

        def fake_processor(*, text, images, padding, return_tensors):
            calls["n"] = len(images)
            calls["pads"] = text[0].count("<|image_pad|>")
            ids = torch.zeros(1, 6, dtype=torch.long)
            return {
                "input_ids": ids,
                "attention_mask": torch.ones(1, 6),
                "pixel_values": torch.zeros(1, 3, 4, 4),
            }

        class FakeQwen(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._p = torch.nn.Parameter(torch.zeros(1))

            def forward(self, **kwargs):
                seq = kwargs["input_ids"].shape[1]
                hs0 = torch.zeros(1, seq, 3)
                return type("Out", (), {"hidden_states": (hs0,)})()

        h, m = grounded_encode_impl(
            "recolor",
            [Image.new("RGB", (64, 64))],
            grounding_px=0,
            mm_processor=fake_processor,
            qwen=FakeQwen(),
            select_layers=(0,),
            prefix_idx=2,
        )
        self.assertEqual(calls["n"], 1)
        self.assertEqual(calls["pads"], 1)
        self.assertEqual(h.shape[1], 4)


if __name__ == "__main__":
    unittest.main()
