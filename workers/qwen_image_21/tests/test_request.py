import base64
import importlib.util
import math
import sys
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

_MISSING = object()
_REQUEST_PATH = Path(__file__).parents[1] / "qwen_image_21_infer" / "request.py"
_SCHEMA_PATH = Path(__file__).parents[1] / "schemas.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name, _MISSING)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


_request = _load("_qwen_image_21_request_under_test", _REQUEST_PATH)
_schemas = _load("_qwen_image_21_schemas_under_test", _SCHEMA_PATH)

RequestError = _request.RequestError
normalize_job_input = _request.normalize_job_input
build_pipe_kwargs = _request.build_pipe_kwargs
OFFICIAL_PRESETS = _request.OFFICIAL_PRESETS
INPUT_SCHEMA = _schemas.INPUT_SCHEMA


def _png_b64(w=32, h=32, color=(255, 0, 0), mode="RGB") -> str:
    buf = BytesIO()
    Image.new(mode, (w, h), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _generate(**overrides):
    payload = {
        "prompt": "a red fox",
        "width": 1024,
        "height": 1024,
        "images": [],
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "num_images": 1,
        "negative_prompt": None,
        "output_resolution": None,
        "seed": None,
    }
    payload.update(overrides)
    raw_keys = set(payload.keys())
    return normalize_job_input(payload, raw_keys=raw_keys)


class NormalizeJobInputTests(unittest.TestCase):
    def test_default_type_is_generate(self):
        out = _generate()
        self.assertEqual(out.type, "image_generate")
        self.assertEqual(out.images, ())
        self.assertFalse(out.size_from_source)
        self.assertEqual(out.output_resolution, 1024)
        self.assertEqual(out.num_inference_steps, 40)
        self.assertEqual(out.guidance_scale, 1.0)

    def test_generate_rejects_images(self):
        with self.assertRaisesRegex(RequestError, "images"):
            _generate(images=[_png_b64()])

    def test_generate_rejects_blank_prompt(self):
        with self.assertRaisesRegex(RequestError, "prompt"):
            _generate(prompt="  ")

    def test_generate_rejects_bad_multiple(self):
        with self.assertRaisesRegex(RequestError, "multiples of 32"):
            _generate(width=1000, height=1024)

    def test_generate_rejects_oversized_area(self):
        with self.assertRaisesRegex(RequestError, "width\\*height"):
            _generate(width=2752, height=2752)

    def test_official_presets_fit(self):
        for name, (width, height) in OFFICIAL_PRESETS.items():
            out = _generate(width=width, height=height)
            self.assertEqual((out.width, out.height), (width, height), name)
            self.assertTrue(INPUT_SCHEMA["width"]["constraints"](width), name)
            self.assertTrue(INPUT_SCHEMA["height"]["constraints"](height), name)

    def test_guidance_above_one_requires_negative_prompt(self):
        with self.assertRaisesRegex(RequestError, "negative_prompt"):
            _generate(guidance_scale=4.0, negative_prompt=None)
        out = _generate(guidance_scale=3.5, negative_prompt="blurry")
        self.assertEqual(out.guidance_scale, 3.5)
        self.assertEqual(out.negative_prompt, "blurry")

    def test_blank_negative_prompt_is_dropped(self):
        out = _generate(negative_prompt="   ")
        self.assertIsNone(out.negative_prompt)

    def test_edit_requires_one_to_ten_images(self):
        with self.assertRaisesRegex(RequestError, "1 to 10"):
            normalize_job_input(
                {
                    "type": "image_edit",
                    "prompt": "make it night",
                    "images": [],
                    "num_images": 1,
                    "width": 1024,
                    "height": 1024,
                },
                raw_keys={"type", "prompt", "images"},
            )
        images = [_png_b64() for _ in range(11)]
        with self.assertRaisesRegex(RequestError, "1 to 10"):
            normalize_job_input(
                {
                    "type": "image_edit",
                    "prompt": "combine them",
                    "images": images,
                    "num_images": 1,
                },
                raw_keys={"type", "prompt", "images"},
            )

    def test_edit_keeps_alpha_and_order(self):
        rgb = _png_b64(color=(0, 128, 0))
        rgba = _png_b64(color=(10, 20, 30, 40), mode="RGBA")
        out = normalize_job_input(
            {
                "type": "image_edit",
                "prompt": "sit them by a fire",
                "images": [rgb, rgba],
                "num_images": 1,
                "num_inference_steps": 40,
                "guidance_scale": 1.0,
            },
            raw_keys={"type", "prompt", "images"},
        )
        self.assertTrue(out.size_from_source)
        self.assertIsNone(out.width)
        self.assertIsNone(out.height)
        self.assertEqual(out.output_resolution, 1024)
        self.assertEqual(out.images[0].mode, "RGB")
        self.assertEqual(out.images[1].mode, "RGBA")
        self.assertEqual(out.images[1].getpixel((0, 0))[3], 40)

    def test_edit_rejects_half_specified_canvas(self):
        with self.assertRaisesRegex(RequestError, "both width and height"):
            normalize_job_input(
                {
                    "type": "image_edit",
                    "prompt": "crop tighter",
                    "images": [_png_b64()],
                    "num_images": 1,
                    "width": 1024,
                    "height": 1024,
                },
                raw_keys={"type", "prompt", "images", "width"},
            )

    def test_edit_explicit_canvas_derives_output_resolution(self):
        out = normalize_job_input(
            {
                "type": "image_edit",
                "prompt": "sunset",
                "images": [_png_b64()],
                "num_images": 1,
                "width": 2048,
                "height": 1024,
                "output_resolution": None,
            },
            raw_keys={"type", "prompt", "images", "width", "height"},
        )
        self.assertEqual((out.width, out.height), (2048, 1024))
        self.assertEqual(out.output_resolution, int(round(math.sqrt(2048 * 1024))))

    def test_edit_honors_explicit_output_resolution(self):
        out = normalize_job_input(
            {
                "type": "image_edit",
                "prompt": "sunset",
                "images": [_png_b64()],
                "num_images": 1,
                "output_resolution": 2048,
            },
            raw_keys={"type", "prompt", "images", "output_resolution"},
        )
        self.assertTrue(out.size_from_source)
        self.assertEqual(out.output_resolution, 2048)

    def test_edit_rejects_num_images_above_one(self):
        with self.assertRaisesRegex(RequestError, "num_images"):
            normalize_job_input(
                {
                    "type": "image_edit",
                    "prompt": "sunset",
                    "images": [_png_b64()],
                    "num_images": 2,
                },
                raw_keys={"type", "prompt", "images"},
            )

    def test_data_url_and_bad_payload(self):
        raw = _png_b64()
        out = normalize_job_input(
            {
                "type": "image_edit",
                "prompt": "ok",
                "images": [f"data:image/png;base64,{raw}"],
                "num_images": 1,
            },
            raw_keys={"type", "prompt", "images"},
        )
        self.assertEqual(len(out.images), 1)
        with self.assertRaisesRegex(RequestError, "base64"):
            normalize_job_input(
                {
                    "type": "image_edit",
                    "prompt": "ok",
                    "images": ["data:image/png,not-base64"],
                    "num_images": 1,
                },
                raw_keys={"type", "prompt", "images"},
            )

    def test_unsupported_type(self):
        with self.assertRaisesRegex(RequestError, "unsupported type"):
            _generate(type="video")


class BuildPipeKwargsTests(unittest.TestCase):
    def test_generate_omits_image_and_pins_cache(self):
        kwargs = build_pipe_kwargs(_generate(seed=7))
        self.assertNotIn("image", kwargs)
        self.assertEqual(kwargs["width"], 1024)
        self.assertEqual(kwargs["height"], 1024)
        self.assertEqual(kwargs["true_cfg_scale"], 1.0)
        self.assertTrue(kwargs["use_kv_cache"])
        self.assertNotIn("negative_prompt", kwargs)
        self.assertNotIn("generator", kwargs)

    def test_edit_passes_every_reference_as_a_list(self):
        out = normalize_job_input(
            {
                "type": "image_edit",
                "prompt": "combine",
                "images": [_png_b64(color=(1, 2, 3)), _png_b64(color=(4, 5, 6))],
                "num_images": 1,
                "guidance_scale": 2.5,
                "negative_prompt": "blur",
            },
            raw_keys={"type", "prompt", "images"},
        )
        kwargs = build_pipe_kwargs(out)
        self.assertNotIn("width", kwargs)
        self.assertNotIn("height", kwargs)
        self.assertEqual(len(kwargs["image"]), 2)
        self.assertEqual(kwargs["negative_prompt"], "blur")
        self.assertEqual(kwargs["true_cfg_scale"], 2.5)
        self.assertEqual(kwargs["output_resolution"], 1024)

    def test_schema_defaults_match_the_contract(self):
        self.assertEqual(INPUT_SCHEMA["num_inference_steps"]["default"], 40)
        self.assertEqual(INPUT_SCHEMA["guidance_scale"]["default"], 1.0)
        self.assertEqual(INPUT_SCHEMA["width"]["default"], 1024)
        self.assertIsNone(INPUT_SCHEMA["output_resolution"]["default"])
        self.assertTrue(INPUT_SCHEMA["output_resolution"]["constraints"](None))
        self.assertFalse(INPUT_SCHEMA["width"]["constraints"](1000))
        self.assertFalse(INPUT_SCHEMA["num_inference_steps"]["constraints"](0))
        self.assertFalse(INPUT_SCHEMA["num_images"]["constraints"](5))


if __name__ == "__main__":
    unittest.main()
