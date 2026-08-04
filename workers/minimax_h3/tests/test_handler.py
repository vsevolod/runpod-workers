"""Unit tests for handler (mocked pipeline / upload; no GPU / no runpod install)."""

from __future__ import annotations

import base64
import os
import unittest
from pathlib import Path
from unittest import mock


def _write_tiny_mp4(path: Path, size: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * size)


def _validated(prompt: str = "a red fox", seed: int = 42) -> dict:
    return {
        "validated_input": {
            "prompt": prompt,
            "width": 864,
            "height": 480,
            "duration": 5.0,
            "seed": seed,
        }
    }


class HandlerTests(unittest.TestCase):
    def setUp(self):
        import handler as handler_mod

        handler_mod._MODELS = None
        self.handler_mod = handler_mod

    def tearDown(self):
        self.handler_mod._MODELS = None

    def _mock_pipe(self, size: int = 128, side_effect=None):
        mock_pipe = mock.MagicMock()
        if side_effect is not None:
            mock_pipe.generate_t2v.side_effect = side_effect
        else:

            def fake_generate(req, output_path):
                p = Path(output_path)
                _write_tiny_mp4(p, size)
                return p

            mock_pipe.generate_t2v.side_effect = fake_generate
        mock_models = mock.MagicMock()
        mock_models.pipe = mock_pipe
        return mock_models

    def test_missing_input(self):
        out = self.handler_mod.handler({"id": "j1"})
        self.assertIn("error", out)
        self.assertIn("input", out["error"].lower())

    def test_valid_base64_no_bucket(self):
        mock_models = self._mock_pipe(128)
        env_clear_keys = (
            "BUCKET_ENDPOINT_URL",
            "BUCKET_ACCESS_KEY_ID",
            "BUCKET_SECRET_ACCESS_KEY",
            "BUCKET_NAME",
        )
        cleared = {k: "" for k in env_clear_keys}
        with mock.patch.dict(os.environ, cleared, clear=False):
            for k in env_clear_keys:
                os.environ.pop(k, None)
            with mock.patch.object(
                self.handler_mod, "get_models", return_value=mock_models
            ):
                with mock.patch.object(
                    self.handler_mod, "_validate", return_value=_validated()
                ):
                    out = self.handler_mod.handler(
                        {
                            "id": "job-b64",
                            "input": {
                                "prompt": "a red fox",
                                "width": 864,
                                "height": 480,
                                "duration": 5.0,
                                "seed": 42,
                            },
                        }
                    )
        self.assertNotIn("error", out)
        self.assertIn("video", out)
        self.assertNotIn("video_base64", out)
        raw = base64.b64decode(out["video"])
        self.assertEqual(len(raw), 128)
        self.assertEqual(out["length"], 124)
        self.assertEqual(out["seed"], 42)
        self.assertEqual(out["fps"], 24)
        self.assertEqual(out["model"], "MiniMaxAI/MiniMax-H3")

    def test_valid_url_with_full_bucket(self):
        mock_models = self._mock_pipe(256)
        env = {
            "BUCKET_ENDPOINT_URL": "https://s3.example",
            "BUCKET_ACCESS_KEY_ID": "ak",
            "BUCKET_SECRET_ACCESS_KEY": "sk",
            "BUCKET_NAME": "mybucket",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.object(
                self.handler_mod, "get_models", return_value=mock_models
            ):
                with mock.patch.object(
                    self.handler_mod,
                    "_validate",
                    return_value=_validated(prompt="waves", seed=1),
                ):
                    with mock.patch.object(
                        self.handler_mod,
                        "_upload_file_to_bucket",
                        return_value="https://example/x.mp4",
                    ) as up:
                        out = self.handler_mod.handler(
                            {
                                "id": "job-url",
                                "input": {"prompt": "waves", "seed": 1},
                            }
                        )
        self.assertNotIn("error", out)
        self.assertEqual(out["video_url"], "https://example/x.mp4")
        self.assertEqual(out["fps"], 24)
        self.assertEqual(out["model"], "MiniMaxAI/MiniMax-H3")
        self.assertNotIn("video", out)
        up.assert_called_once()
        self.assertEqual(up.call_args.kwargs.get("bucket_name"), "mybucket")

    def test_incomplete_bucket_large_file_errors(self):
        mock_models = self._mock_pipe(7_000_001)
        with mock.patch.dict(
            os.environ,
            {
                "BUCKET_ENDPOINT_URL": "https://s3.example",
                "BUCKET_ACCESS_KEY_ID": "",
                "BUCKET_SECRET_ACCESS_KEY": "",
                "BUCKET_NAME": "",
            },
            clear=False,
        ):
            for k in (
                "BUCKET_ACCESS_KEY_ID",
                "BUCKET_SECRET_ACCESS_KEY",
                "BUCKET_NAME",
            ):
                os.environ.pop(k, None)
            with mock.patch.object(
                self.handler_mod, "get_models", return_value=mock_models
            ):
                with mock.patch.object(
                    self.handler_mod,
                    "_validate",
                    return_value=_validated(prompt="big", seed=2),
                ):
                    out = self.handler_mod.handler(
                        {"id": "job-big", "input": {"prompt": "big", "seed": 2}}
                    )
        self.assertIn("error", out)
        self.assertNotIn("video_url", out)

    def test_oom_sets_refresh_worker(self):
        class OutOfMemoryError(Exception):
            pass

        mock_models = self._mock_pipe(side_effect=OutOfMemoryError("cuda oom"))
        with mock.patch.object(
            self.handler_mod, "get_models", return_value=mock_models
        ):
            with mock.patch.object(
                self.handler_mod,
                "_validate",
                return_value=_validated(prompt="oom", seed=3),
            ):
                out = self.handler_mod.handler(
                    {"id": "job-oom", "input": {"prompt": "oom", "seed": 3}}
                )
        self.assertIn("error", out)
        self.assertTrue(out.get("refresh_worker"))
        self.assertEqual(out["error"], "CUDA out of memory")

    def test_upload_local_path_rejected(self):
        mock_models = self._mock_pipe(64)
        env = {
            "BUCKET_ENDPOINT_URL": "https://s3.example",
            "BUCKET_ACCESS_KEY_ID": "ak",
            "BUCKET_SECRET_ACCESS_KEY": "sk",
            "BUCKET_NAME": "mybucket",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.object(
                self.handler_mod, "get_models", return_value=mock_models
            ):
                with mock.patch.object(
                    self.handler_mod,
                    "_validate",
                    return_value=_validated(prompt="path", seed=4),
                ):
                    with mock.patch.object(
                        self.handler_mod,
                        "_upload_file_to_bucket",
                        return_value="local_upload/foo.mp4",
                    ):
                        out = self.handler_mod.handler(
                            {
                                "id": "job-local",
                                "input": {"prompt": "path", "seed": 4},
                            }
                        )
        self.assertIn("error", out)
        self.assertIn("non-URL", out["error"])
        self.assertNotIn("video_url", out)


if __name__ == "__main__":
    unittest.main()
