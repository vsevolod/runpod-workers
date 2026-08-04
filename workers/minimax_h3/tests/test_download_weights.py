"""Unit tests for download_weights (no network)."""

from __future__ import annotations

import unittest
from unittest import mock

from download_weights import ALLOW_PATTERNS_HYBRID_TE_VAE, ALLOW_PATTERNS_T2VA, main


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


class TestHybridPack(unittest.TestCase):
    def test_hybrid_dry_run(self):
        self.assertEqual(
            main(["--output", "/tmp/x", "--pack", "hybrid_spike", "--dry-run"]), 0
        )

    def test_hybrid_downloads_non_pruned_dit(self):
        with mock.patch(
            "download_weights._hf_hub_download", return_value="/tmp/dit"
        ) as hd:
            with mock.patch(
                "download_weights._snapshot_download", return_value="/tmp/off"
            ) as sd:
                with mock.patch("pathlib.Path.mkdir"):
                    main(["--output", "/tmp/x", "--pack", "hybrid_spike"])
        self.assertIn(
            "minimax_h3_fl2va_int8_convrot.safetensors", str(hd.call_args)
        )
        self.assertNotIn("pruned", str(hd.call_args))
        patterns = sd.call_args.kwargs["allow_patterns"]
        joined = " ".join(patterns)
        self.assertIn("transformer/config.json", joined)
        # must not pull official ~66GB DiT weight shards
        self.assertNotIn("transformer/*.safetensors", joined)
        self.assertNotIn("transformer/**", joined)

    def test_hybrid_patterns_constant(self):
        joined = " ".join(ALLOW_PATTERNS_HYBRID_TE_VAE)
        self.assertIn("transformer/config.json", joined)
        self.assertNotIn("transformer/**", joined)
        self.assertNotIn("transformer/*.safetensors", joined)


if __name__ == "__main__":
    unittest.main()
