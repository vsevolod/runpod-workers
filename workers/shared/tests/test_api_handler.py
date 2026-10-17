"""Exercise the shared handler with real image delivery and inference boundaries."""

import base64
import importlib
from io import BytesIO
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[2]))


class APIHandlerTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("shared.api")
        except ModuleNotFoundError:
            self.fail("The shared worker API is not implemented")
        self.calls = []

    def handle(self, job, **overrides):
        def normalize(values, *, raw_keys):
            return self.api.normalize_common_fields(values, self.api.KREA_API, raw_keys=raw_keys)
        def infer(norm, seed, loras):
            self.calls.append((norm, seed, loras))
            return self.api.GenerationResult([Image.new("RGB", (64, 48), (200, 50, 20))], [], {"mu": 1.15})
        return self.api.handle_job(
            job, profile=self.api.KREA_API, normalize=normalize,
            infer=overrides.pop("infer", infer), logger=logging.getLogger(__name__), **overrides,
        )

    def test_response_contains_actual_size_sampling_options_and_decodable_png(self):
        with patch.dict("os.environ", {}, clear=True):
            output = self.handle({"input": {"prompt": "fox", "seed": 0, "num_inference_steps": 25, "guidance_scale": 2.5}})
        self.assertEqual(output["seed"], 0)
        self.assertEqual((output["width"], output["height"]), (64, 48))
        self.assertEqual(output["num_inference_steps"], 25)
        self.assertEqual(output["guidance_scale"], 2.5)
        self.assertEqual(output["sampler"], "euler")
        self.assertEqual(output["model_options"], {"mu": 1.15})
        raw = base64.b64decode(output["images"][0].split(",", 1)[1])
        image = Image.open(BytesIO(raw))
        self.assertEqual(image.size, (64, 48))
        self.assertEqual(image.getpixel((0, 0)), (200, 50, 20))

    def test_seed_is_random_only_when_omitted_or_null(self):
        for fields in ({}, {"seed": None}):
            with patch("os.urandom", return_value=b"\x00\x00\x00\x07"):
                self.assertEqual(self.handle({"input": {"prompt": "fox", **fields}})["seed"], 7)
        with patch("os.urandom", side_effect=AssertionError("Explicit seed replaced")):
            self.assertEqual(self.handle({"input": {"prompt": "fox", "seed": 0}})["seed"], 0)

    def test_invalid_inputs_never_reach_inference(self):
        for job in ({"input": {"prompt": "fox", "sampler": "heun"}}, {"input": None}, {"input": {"prompt": " "}}):
            self.assertIn("error", self.handle(job))
        self.assertEqual(self.calls, [])

    def test_upload_failure_falls_back_to_png_and_inference_errors_keep_restart_policy(self):
        def broken_delivery(*args): raise RuntimeError("upload unavailable")
        output = self.handle({"input": {"prompt": "fox"}}, deliver=broken_delivery)
        self.assertTrue(output["images"][0].startswith("data:image/png;base64,"))
        for failure, restart in [(FileNotFoundError("weights"), False), (RuntimeError("bad inference"), True)]:
            def broken_inference(*args): raise failure
            output = self.handle({"input": {"prompt": "fox"}}, infer=broken_inference)
            self.assertIn("error", output)
            self.assertEqual(output.get("refresh_worker", False), restart)

    def test_bucket_delivery_uploads_valid_png_and_cleans_files_on_success_or_error(self):
        from shared.api.images import save_and_upload_images
        for fail in (False, True):
            paths = []
            def upload(job_id, path):
                paths.append(Path(path))
                self.assertEqual(job_id, "request-id")
                with Image.open(path) as image:
                    self.assertEqual((image.format, image.size), ("PNG", (64, 48)))
                if fail:
                    raise RuntimeError("upload unavailable")
                return "https://storage.example/result.png"
            with patch.dict("os.environ", {"BUCKET_ENDPOINT_URL": "https://storage.example"}), patch.dict(
                sys.modules, {"runpod.serverless.utils": SimpleNamespace(rp_upload=SimpleNamespace(upload_image=upload))}
            ):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "upload unavailable"):
                        save_and_upload_images([Image.new("RGB", (64, 48))], "request-id")
                else:
                    self.assertEqual(save_and_upload_images([Image.new("RGB", (64, 48))], "request-id"), ["https://storage.example/result.png"])
            self.assertEqual(len(paths), 1)
            self.assertFalse(paths[0].exists())
            self.assertFalse(paths[0].parent.exists())


if __name__ == "__main__":
    unittest.main()
