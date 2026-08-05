"""BUCKET_* matrix + SaveVideo path resolution (no live Comfy)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import handler  # noqa: E402


class TestBucketState(unittest.TestCase):
    def test_none(self):
        self.assertEqual(handler.bucket_state({}), "none")

    def test_full(self):
        env = {k: "x" for k in handler.BUCKET_KEYS}
        self.assertEqual(handler.bucket_state(env), "full")

    def test_partial(self):
        env = {"BUCKET_NAME": "n", "BUCKET_ENDPOINT_URL": "http://x"}
        self.assertEqual(handler.bucket_state(env), "partial")


class TestPathFromSaveVideo(unittest.TestCase):
    def test_resolves_images_entry(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            sub = out / "video" / "MiniMax_H3"
            sub.mkdir(parents=True)
            mp4 = sub / "MiniMax_H3_00001_.mp4"
            mp4.write_bytes(b"\x00\x00fake")
            history = {
                "outputs": {
                    "92": {
                        "images": [
                            {
                                "filename": "MiniMax_H3_00001_.mp4",
                                "subfolder": "video/MiniMax_H3",
                                "type": "output",
                            }
                        ],
                        "animated": [True],
                    }
                }
            }
            got = handler.path_from_savevideo(history, out)
            self.assertEqual(got, mp4)

    def test_missing_node_raises(self):
        with self.assertRaises(RuntimeError):
            handler.path_from_savevideo({"outputs": {}}, Path("/tmp"))

    def test_no_images_raises_no_glob(self):
        with self.assertRaises(RuntimeError) as ctx:
            handler.path_from_savevideo(
                {"outputs": {"92": {"gifs": []}}}, Path("/tmp")
            )
        self.assertIn("images", str(ctx.exception).lower())


class TestNormalizeInput(unittest.TestCase):
    def test_seed_negative_randomizes(self):
        p = handler.normalize_input(
            {"prompt": "hi", "width": 864, "height": 480, "seed": -1}
        )
        self.assertGreaterEqual(p["seed"], 0)

    def test_defaults(self):
        p = handler.normalize_input({"prompt": "hi"})
        self.assertEqual(p["width"], 864)
        self.assertEqual(p["height"], 480)


class TestDeliverVideo(unittest.TestCase):
    def test_inline_when_no_bucket(self):
        with tempfile.TemporaryDirectory() as td:
            mp4 = Path(td) / "a.mp4"
            mp4.write_bytes(b"abc123")
            with mock.patch.dict("os.environ", {}, clear=False):
                for k in handler.BUCKET_KEYS:
                    if k in __import__("os").environ:
                        del __import__("os").environ[k]
                # ensure none mode
                with mock.patch.object(handler, "bucket_state", return_value="none"):
                    with mock.patch.object(handler, "MAX_INLINE_VIDEO_BYTES", 100):
                        out = handler.deliver_video(mp4, "job1")
            self.assertEqual(out["delivery"], "base64")
            self.assertTrue(out["video"].startswith("data:video/mp4;base64,"))

    def test_inline_too_large(self):
        with tempfile.TemporaryDirectory() as td:
            mp4 = Path(td) / "a.mp4"
            mp4.write_bytes(b"x" * 50)
            with mock.patch.object(handler, "bucket_state", return_value="none"):
                with mock.patch.object(handler, "MAX_INLINE_VIDEO_BYTES", 10):
                    with self.assertRaises(RuntimeError):
                        handler.deliver_video(mp4, "job1")


if __name__ == "__main__":
    unittest.main()
