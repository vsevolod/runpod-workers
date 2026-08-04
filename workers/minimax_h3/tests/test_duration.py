"""Unit tests for h3_infer.duration."""

from __future__ import annotations

import unittest

from h3_infer.duration import (
    MAX_DURATION_SEC,
    MIN_DURATION_SEC,
    output_duration_sec,
    snap_num_frames,
    validate_requested_duration,
)


class TestSnapNumFrames(unittest.TestCase):
    def test_five_seconds_is_124(self):
        self.assertEqual(snap_num_frames(5.0), 124)

    def test_max_duration_is_345(self):
        self.assertEqual(snap_num_frames(14.375), 345)

    def test_form_17n_plus_5(self):
        for d in (5.0, 6.0, 8.5, 10.0, 12.0, 14.375):
            n = snap_num_frames(d)
            self.assertEqual(n % 17, 5, msg=f"duration={d} frames={n}")


class TestValidateRequestedDuration(unittest.TestCase):
    def test_accepts_bounds(self):
        self.assertEqual(validate_requested_duration(5.0), 5.0)
        self.assertEqual(validate_requested_duration(14.375), 14.375)

    def test_rejects_below_min(self):
        with self.assertRaises(ValueError):
            validate_requested_duration(4.99)

    def test_rejects_above_max(self):
        with self.assertRaises(ValueError):
            validate_requested_duration(15.0)

    def test_constants_match_plan(self):
        self.assertEqual(MIN_DURATION_SEC, 5.0)
        self.assertEqual(MAX_DURATION_SEC, 14.375)


class TestOutputDuration(unittest.TestCase):
    def test_124_frames(self):
        self.assertAlmostEqual(output_duration_sec(124), 124 / 24)

    def test_345_frames(self):
        self.assertAlmostEqual(output_duration_sec(345), 14.375)


if __name__ == "__main__":
    unittest.main()
