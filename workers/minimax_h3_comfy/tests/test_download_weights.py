"""download_weights targets exactly four files."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import download_weights as dw  # noqa: E402


class TestDownloadWeights(unittest.TestCase):
    def test_exactly_four_targets(self):
        with tempfile.TemporaryDirectory() as td:
            targets = dw.expected_targets(Path(td))
            self.assertEqual(len(targets), 4)
            names = [p.name for p in targets]
            self.assertEqual(
                names,
                [
                    "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                    "minimax_h3_video_vae_fp16.safetensors",
                    "minimax_h3_audio_vae_fp32.safetensors",
                ],
            )

    def test_dry_run_exit_zero(self):
        with tempfile.TemporaryDirectory() as td:
            rc = dw.main(["--output", td, "--dry-run"])
            self.assertEqual(rc, 0)

    def test_download_calls_hf_four_times(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)

            def fake_dl(repo_id, filename, dest_dir, token):
                rel = Path(filename)
                path = Path(dest_dir) / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
                return path

            with mock.patch.object(dw, "download_one", side_effect=fake_dl) as m:
                rc = dw.main(["--output", str(out)])
            self.assertEqual(rc, 0)
            self.assertEqual(m.call_count, 4)
            for p in dw.expected_targets(out):
                self.assertTrue(p.is_file())


if __name__ == "__main__":
    unittest.main()
