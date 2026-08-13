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
        self.assertIsNone(p["first_image"])
        self.assertIsNone(p["last_image"])

    def test_first_image_accepted(self):
        p = handler.normalize_input(
            {"prompt": "hi", "first_image": "data:image/png;base64,abc"}
        )
        self.assertEqual(p["first_image"], "data:image/png;base64,abc")

    def test_last_without_first_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            handler.normalize_input({"prompt": "hi", "last_image": "x"})
        self.assertIn("first_image", str(ctx.exception))

    def test_empty_first_image_rejected(self):
        with self.assertRaises(ValueError):
            handler.normalize_input({"prompt": "hi", "first_image": "  "})


class TestRegionFromEndpoint(unittest.TestCase):
    def test_runpod_s3api_host(self):
        self.assertEqual(
            handler.region_from_endpoint("https://s3api-eu-ro-1.runpod.io/"),
            "EU-RO-1",
        )

    def test_runpod_s3api_no_scheme(self):
        self.assertEqual(
            handler.region_from_endpoint("s3api-us-ks-2.runpod.io"),
            "US-KS-2",
        )

    def test_unknown_endpoint(self):
        self.assertIsNone(
            handler.region_from_endpoint("https://account.r2.cloudflarestorage.com")
        )

    def test_empty(self):
        self.assertIsNone(handler.region_from_endpoint(""))
        self.assertIsNone(handler.region_from_endpoint("   "))


class TestBucketRegion(unittest.TestCase):
    def test_explicit_wins(self):
        env = {
            "BUCKET_REGION": "US-KS-2",
            "BUCKET_ENDPOINT_URL": "https://s3api-eu-ro-1.runpod.io/",
        }
        self.assertEqual(handler.bucket_region(env), "US-KS-2")

    def test_parsed_from_endpoint(self):
        env = {"BUCKET_ENDPOINT_URL": "https://s3api-eu-ro-1.runpod.io/"}
        self.assertEqual(handler.bucket_region(env), "EU-RO-1")

    def test_missing(self):
        self.assertIsNone(handler.bucket_region({}))


class TestDeliverVideo(unittest.TestCase):
    def test_s3_returns_bucket_and_key(self):
        with tempfile.TemporaryDirectory() as td:
            mp4 = Path(td) / "MiniMax_H3_00001_.mp4"
            mp4.write_bytes(b"mp4bytes")
            with mock.patch.object(handler, "bucket_state", return_value="full"):
                with mock.patch.object(
                    handler,
                    "_upload_video",
                    return_value=("vol_abc", "job1/MiniMax_H3_00001_.mp4"),
                ) as upload:
                    out = handler.deliver_video(mp4, "job1")
        self.assertEqual(out["delivery"], "s3")
        self.assertEqual(out["bucket"], "vol_abc")
        self.assertEqual(out["key"], "job1/MiniMax_H3_00001_.mp4")
        self.assertEqual(out["bytes"], 8)
        self.assertNotIn("video_url", out)
        self.assertNotIn("video", out)
        upload.assert_called_once_with("job1", mp4)

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


class TestUploadVideo(unittest.TestCase):
    def test_puts_object_and_skips_presign(self):
        with tempfile.TemporaryDirectory() as td:
            mp4 = Path(td) / "clip.mp4"
            mp4.write_bytes(b"data")
            fake_client = mock.Mock()
            env = {
                "BUCKET_ENDPOINT_URL": "https://s3api-eu-ro-1.runpod.io/",
                "BUCKET_ACCESS_KEY_ID": "ak",
                "BUCKET_SECRET_ACCESS_KEY": "sk",
                "BUCKET_NAME": "vol_abc",
            }
            with mock.patch.dict("os.environ", env, clear=False):
                with mock.patch("boto3.client", return_value=fake_client) as make:
                    bucket, key = handler._upload_video("job9", mp4)
        self.assertEqual(bucket, "vol_abc")
        self.assertEqual(key, "job9/clip.mp4")
        make.assert_called_once()
        kwargs = make.call_args.kwargs
        self.assertEqual(kwargs["endpoint_url"], "https://s3api-eu-ro-1.runpod.io/")
        self.assertEqual(kwargs["region_name"], "EU-RO-1")
        fake_client.upload_file.assert_called_once_with(
            str(mp4),
            "vol_abc",
            "job9/clip.mp4",
            ExtraArgs={"ContentType": "video/mp4"},
        )
        fake_client.generate_presigned_url.assert_not_called()

    def test_incomplete_env_raises(self):
        with tempfile.TemporaryDirectory() as td:
            mp4 = Path(td) / "clip.mp4"
            mp4.write_bytes(b"data")
            env = {k: "" for k in handler.BUCKET_KEYS}
            env["BUCKET_NAME"] = "vol_abc"
            with mock.patch.dict("os.environ", env, clear=False):
                with self.assertRaises(RuntimeError):
                    handler._upload_video("job9", mp4)


if __name__ == "__main__":
    unittest.main()
