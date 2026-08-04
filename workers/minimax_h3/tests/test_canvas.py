"""Unit tests for h3_infer.canvas (packing.py parity)."""

from __future__ import annotations

import unittest

from h3_infer.canvas import resolve_canvas_size, validate_canvas


class TestResolveCanvas(unittest.TestCase):
    def test_16_9(self):
        h, w = resolve_canvas_size(16, 9)
        self.assertEqual((h, w), (768, 1344))

    def test_9_16(self):
        h, w = resolve_canvas_size(9, 16)
        self.assertEqual((h, w), (1344, 768))

    def test_1_1(self):
        h, w = resolve_canvas_size(1, 1)
        self.assertEqual((h, w), (768, 768))

    def test_4_1(self):
        h, w = resolve_canvas_size(4, 1)
        self.assertEqual((h, w), (512, 2016))

    def test_1_4(self):
        h, w = resolve_canvas_size(1, 4)
        self.assertEqual((h, w), (2016, 512))


class TestValidateCanvas(unittest.TestCase):
    def test_accepts_preview(self):
        validate_canvas(864, 480)

    def test_accepts_4_1_official(self):
        validate_canvas(2016, 512)

    def test_accepts_1_4_official(self):
        validate_canvas(512, 2016)

    def test_rejects_not_multiple_32(self):
        with self.assertRaises(ValueError):
            validate_canvas(865, 480)

    def test_rejects_over_nominal_area(self):
        # larger than 16:9 nominal 1344x768 area
        with self.assertRaises(ValueError):
            validate_canvas(1920, 1088)


if __name__ == "__main__":
    unittest.main()
