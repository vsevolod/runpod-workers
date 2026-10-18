import base64
import importlib.util
import sys
import unittest
from unittest.mock import patch
from io import BytesIO
from pathlib import Path

from PIL import Image

_MISSING = object()
_PACKAGE_ROOT = Path(__file__).parents[1] / "krea2_infer"
_REQUEST_MODULE_NAME = "_krea2_request_contract_tests"


def _load_request_module():
    """Load request.py without importing krea2_infer package (avoids heavy deps)."""
    spec = importlib.util.spec_from_file_location(
        _REQUEST_MODULE_NAME,
        _PACKAGE_ROOT / "request.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the request contract module")

    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(_REQUEST_MODULE_NAME, _MISSING)
    sys.modules[_REQUEST_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is _MISSING:
            sys.modules.pop(_REQUEST_MODULE_NAME, None)
        else:
            sys.modules[_REQUEST_MODULE_NAME] = previous
    return module


_request = _load_request_module()
RequestError = _request.RequestError
normalize_job_input = _request.normalize_job_input


def _png_b64(w=32, h=32, color=(255, 0, 0)) -> str:
    buf = BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class NormalizeJobInputTests(unittest.TestCase):
    def test_shared_max_budget_uses_quant_profile_and_keeps_full_output_area(self):
        from shared import megapixels
        from shared.api import KREA_API, parse_job_input
        raw = {"type": "image_edit", "prompt": "combine", "images": [_png_b64(2048, 2048)] * 2,
               "reference_megapixels": "max"}
        parsed = parse_job_input({"input": raw}, KREA_API)
        with patch.dict("os.environ", {"DIT_QUANT": "int8"}), patch.object(
            megapixels, "detect_vram_gib", return_value=47.5
        ), patch.dict(megapixels.MEGAPIXEL_LIMITS, {"krea2.int8_convrot": {0: 1., 48: 2.}}):
            norm = normalize_job_input(parsed.values, raw_keys=parsed.raw_keys, quant_mode="int8_convrot")
        self.assertEqual(norm.reference_megapixels, 2.)
        self.assertEqual([im.size for im in norm.images], [(1000, 1000)] * 2)
        self.assertEqual((norm.width, norm.height), (1408, 1408))
        self.assertTrue(norm.size_from_source)

    def test_default_type_is_generate(self):
        out = normalize_job_input(
            {"prompt": "cat", "width": 1024, "height": 1024, "images": []},
            raw_keys={"prompt", "width", "height"},
        )
        self.assertEqual(out.type, "image_generate")
        self.assertEqual(out.images, ())
        self.assertFalse(out.size_from_source)

    def test_generate_rejects_nonempty_images(self):
        with self.assertRaisesRegex(RequestError, "images"):
            normalize_job_input(
                {
                    "type": "image_generate",
                    "prompt": "cat",
                    "images": [_png_b64()],
                    "width": 1024,
                    "height": 1024,
                },
                raw_keys={"type", "prompt", "images", "width", "height"},
            )

    def test_edit_requires_one_or_two_images(self):
        with self.assertRaisesRegex(RequestError, r"1 or 2"):
            normalize_job_input(
                {"type": "image_edit", "prompt": "make it night", "images": []},
                raw_keys={"type", "prompt", "images"},
            )
        with self.assertRaisesRegex(RequestError, r"1 or 2"):
            normalize_job_input(
                {
                    "type": "image_edit",
                    "prompt": "make it night",
                    "images": [_png_b64(), _png_b64(), _png_b64()],
                },
                raw_keys={"type", "prompt", "images"},
            )

    def test_edit_accepts_two_images_different_sizes(self):
        a = _png_b64(64, 48, color=(255, 0, 0))
        b = _png_b64(100, 200, color=(0, 255, 0))
        out = normalize_job_input(
            {
                "type": "image_edit",
                "prompt": "put person next to tractor",
                "images": [a, b],
                "width": 1024,
                "height": 1024,
                "grounding_px": 768,
                "ref_boost": 4.0,
                "fit_mode": "fit",
                "num_images": 1,
            },
            raw_keys={"type", "prompt", "images", "width", "height"},
        )
        self.assertEqual(len(out.images), 2)
        self.assertEqual(out.images[0].size, (64, 48))
        self.assertEqual(out.images[1].size, (100, 200))
        self.assertFalse(out.size_from_source)
        self.assertEqual(out.width, 1024)
        self.assertEqual(out.height, 1024)

    def test_edit_size_from_source_ignores_second_image_dims(self):
        """Canvas primary is images[0]; second image size does not set WH."""
        a = _png_b64(80, 60)
        b = _png_b64(400, 300)
        out = normalize_job_input(
            {
                "type": "image_edit",
                "prompt": "restage",
                "images": [a, b],
                "width": 1024,
                "height": 1024,
                "grounding_px": 768,
                "ref_boost": 1.0,
                "fit_mode": "fit",
                "num_images": 1,
            },
            raw_keys={"type", "prompt", "images"},
        )
        self.assertTrue(out.size_from_source)
        self.assertEqual((out.width, out.height), (1152, 864))
        self.assertEqual(len(out.images), 2)
        self.assertEqual(out.images[0].size, (80, 60))

    def test_edit_accepts_data_url_and_defaults(self):
        raw = _png_b64(64, 48)
        out = normalize_job_input(
            {
                "type": "image_edit",
                "prompt": "recolor the car matte black",
                "images": [f"data:image/png;base64,{raw}"],
                "width": 1024,
                "height": 1024,
                "grounding_px": 768,
                "ref_boost": 1.0,
                "fit_mode": "fit",
                "num_images": 1,
            },
            raw_keys={"type", "prompt", "images"},  # width/height NOT in raw
        )
        self.assertEqual(out.type, "image_edit")
        self.assertEqual(len(out.images), 1)
        self.assertEqual(out.images[0].size, (64, 48))
        self.assertTrue(out.size_from_source)
        self.assertEqual(out.grounding_px, 768)
        self.assertEqual(out.ref_boost, 1.0)
        self.assertEqual(out.fit_mode, "fit")

    def test_edit_derives_size_when_width_height_omitted_before_validation(self):
        """Schema may inject 1024x1024; raw_keys prove client omitted size."""
        raw = _png_b64(800, 600)
        validated = {
            "type": "image_edit",
            "prompt": "make it night",
            "images": [raw],
            "width": 1024,
            "height": 1024,
            "grounding_px": 768,
            "ref_boost": 1.0,
            "fit_mode": "fit",
            "num_images": 1,
            "num_inference_steps": 8,
            "guidance_scale": 0.0,
            "mu": 1.15,
            "seed": None,
            "negative_prompt": None,
            "loras": [],
        }
        out = normalize_job_input(
            validated,
            raw_keys={"type", "prompt", "images"},
        )
        self.assertTrue(out.size_from_source)
        self.assertEqual((out.width, out.height), (1152, 864))

    def test_edit_honors_explicit_width_height_in_raw(self):
        raw = _png_b64(800, 600)
        out = normalize_job_input(
            {
                "type": "image_edit",
                "prompt": "x",
                "images": [raw],
                "width": 1024,
                "height": 576,
                "grounding_px": 768,
                "ref_boost": 4.0,
                "fit_mode": "fit",
                "num_images": 1,
            },
            raw_keys={"type", "prompt", "images", "width", "height"},
        )
        self.assertFalse(out.size_from_source)
        self.assertEqual(out.width, 1024)
        self.assertEqual(out.height, 576)

    def test_edit_rejects_num_images_gt_1(self):
        with self.assertRaises(RequestError):
            normalize_job_input(
                {
                    "type": "image_edit",
                    "prompt": "x",
                    "images": [_png_b64()],
                    "num_images": 2,
                    "width": 1024,
                    "height": 1024,
                },
                raw_keys={"type", "prompt", "images", "num_images"},
            )

    def test_unknown_type(self):
        with self.assertRaisesRegex(RequestError, "type"):
            normalize_job_input(
                {"type": "video", "prompt": "x", "images": []},
                raw_keys={"type", "prompt"},
            )


if __name__ == "__main__":
    unittest.main()
