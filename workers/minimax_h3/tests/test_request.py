"""Unit tests for h3_infer.request."""

from __future__ import annotations

import unittest

from h3_infer.request import RequestError, T2VRequest, normalize_t2v_input


class TestNormalizeT2VInput(unittest.TestCase):
    def test_defaults_five_sec_preview(self):
        req = normalize_t2v_input({"prompt": "a cat walks", "seed": 42})
        self.assertIsInstance(req, T2VRequest)
        self.assertEqual(req.width, 864)
        self.assertEqual(req.height, 480)
        self.assertEqual(req.requested_duration, 5.0)
        self.assertEqual(req.length, 124)
        self.assertEqual(req.seed, 42)
        self.assertAlmostEqual(req.output_duration, 124 / 24)

    def test_empty_prompt_rejected(self):
        with self.assertRaises(RequestError):
            normalize_t2v_input({"prompt": "  "})

    def test_duration_15_rejected(self):
        with self.assertRaises(RequestError):
            normalize_t2v_input({"prompt": "x", "duration": 15})

    def test_non_multiple_32_rejected(self):
        with self.assertRaises(RequestError):
            normalize_t2v_input({"prompt": "x", "width": 865, "height": 480})

    def test_accepts_2016x512(self):
        req = normalize_t2v_input(
            {"prompt": "wide shot", "width": 2016, "height": 512, "seed": 1}
        )
        self.assertEqual(req.width, 2016)
        self.assertEqual(req.height, 512)

    def test_seed_minus_one_randomizes(self):
        a = normalize_t2v_input({"prompt": "a", "seed": -1})
        b = normalize_t2v_input({"prompt": "a", "seed": -1})
        self.assertIsInstance(a.seed, int)
        self.assertIsInstance(b.seed, int)
        # Extremely unlikely to collide if both random; allow equality but both >= 0
        self.assertGreaterEqual(a.seed, 0)
        self.assertGreaterEqual(b.seed, 0)


if __name__ == "__main__":
    unittest.main()
