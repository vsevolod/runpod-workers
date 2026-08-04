"""Unit tests for download_weights (no network)."""

from __future__ import annotations

import unittest
from unittest import mock

from download_weights import ALLOW_PATTERNS_T2VA, main


class TestPatterns(unittest.TestCase):
    def test_allow_includes_transformer(self):
        self.assertTrue(any(p.startswith("transformer") for p in ALLOW_PATTERNS_T2VA))

    def test_allow_excludes_ref_by_omission(self):
        joined = " ".join(ALLOW_PATTERNS_T2VA)
        self.assertNotIn("transformer_ref", joined)
        self.assertNotIn("Ref2VA", joined)


class TestCliDryRun(unittest.TestCase):
    def test_dry_run_exit_0(self):
        rc = main(["--output", "/tmp/x", "--dry-run"])
        self.assertEqual(rc, 0)

    def test_real_path_calls_snapshot(self):
        with mock.patch(
            "download_weights._snapshot_download", return_value="/tmp/x"
        ) as sd:
            with mock.patch("pathlib.Path.mkdir"):
                rc = main(["--output", "/tmp/x"])
        self.assertEqual(rc, 0)
        sd.assert_called_once()
        kwargs = sd.call_args.kwargs
        self.assertIn("allow_patterns", kwargs)
        self.assertTrue(
            any(p.startswith("transformer") for p in kwargs["allow_patterns"])
        )


if __name__ == "__main__":
    unittest.main()
