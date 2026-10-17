import base64
import importlib.util
import math
import sys
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))

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


def _png_b64(w=256, h=256, color=(255, 0, 0), mode="RGB") -> str:
    buf = BytesIO()
    Image.new(mode, (w, h), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _generate(**overrides):
    payload = {
        "prompt": "a red fox",
        "width": 1024,
        "height": 1024,
        "images": [],
        "num_inference_steps": 8,
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
    def test_base_defaults_to_forty_steps_even_with_schema_defaults(self):
        payload = {"prompt": "fox", "model_variant": "base", "width": 1024, "height": 1024,
                   "num_inference_steps": 8}
        out = normalize_job_input(payload, raw_keys={"prompt", "model_variant"})
        self.assertEqual(out.model_variant, "base")
        self.assertEqual(out.num_inference_steps, 40)

    def test_explicit_steps_override_both_variant_defaults(self):
        for variant, steps in [("base", 25), ("turbo", 12)]:
            with self.subTest(variant=variant):
                out = _generate(model_variant=variant, num_inference_steps=steps)
                self.assertEqual(out.model_variant, variant)
                self.assertEqual(out.num_inference_steps, steps)
                self.assertEqual(len(build_pipe_kwargs(out)["sigmas"]), steps)
                self.assertEqual(out.scheduler, "simple")
        with self.assertRaisesRegex(RequestError, "model_variant"):
            _generate(model_variant="other")

    def test_base_passes_comfy_simple_sigmas_and_turbo_uses_checkpoint_schedule(self):
        base = build_pipe_kwargs(_generate(model_variant="base", num_inference_steps=40))
        self.assertEqual(len(base["sigmas"]), 40)
        self.assertEqual(base["sigmas"][0], 1.0)
        self.assertGreater(base["sigmas"][-1], 0)
        self.assertNotIn("sigmas", build_pipe_kwargs(_generate()))

    def test_default_type_is_generate(self):
        out = _generate()
        self.assertEqual(out.type, "image_generate")
        self.assertEqual(out.images, ())
        self.assertFalse(out.size_from_source)
        self.assertEqual(out.output_resolution, 1024)
        self.assertEqual(out.num_inference_steps, 8)
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

    def test_guidance_above_one_uses_empty_unconditional_prompt_when_omitted(self):
        for negative in (None, "", "   "):
            out = _generate(guidance_scale=4.0, negative_prompt=negative)
            self.assertEqual(out.negative_prompt, "")
            self.assertEqual(build_pipe_kwargs(out)["negative_prompt"], "")
            self.assertEqual(build_pipe_kwargs(out)["true_cfg_scale"], 4.0)
        out = _generate(guidance_scale=3.5, negative_prompt="blurry")
        self.assertEqual(out.guidance_scale, 3.5)
        self.assertEqual(out.negative_prompt, "blurry")

    def test_samplers_are_validated_before_inference_and_default_to_euler(self):
        self.assertEqual(_generate().sampler, "euler")
        self.assertEqual(_generate().scheduler, "checkpoint")
        for sampler in ("euler", "heun", "er_sde"):
            self.assertEqual(_generate(sampler=sampler).sampler, sampler)
            self.assertTrue(INPUT_SCHEMA["sampler"]["constraints"](sampler))
        for sampler in ("unknown", None, [], "Euler"):
            with self.assertRaisesRegex(RequestError, "sampler"):
                _generate(sampler=sampler)

    def test_invalid_guidance_is_a_client_error(self):
        for guidance in (float("nan"), float("inf"), -1, "bad", True, []):
            with self.subTest(guidance=guidance), self.assertRaisesRegex(RequestError, "guidance_scale"):
                _generate(guidance_scale=guidance)

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
                "guidance_scale": 1.0,
            },
            raw_keys={"type", "prompt", "images"},
        )
        self.assertTrue(out.size_from_source)
        self.assertEqual((out.width, out.height), (256, 256))
        self.assertEqual(out.output_resolution, 256)
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

    def test_edit_uses_last_reference_native_canvas_and_keeps_distinct_ref_sizes(self):
        payload = {
            "type": "image_edit", "prompt": "combine",
            "images": [_png_b64(1536, 1024), _png_b64(801, 1191, mode="RGBA")],
        }
        out = normalize_job_input(payload, raw_keys=set(payload))
        self.assertEqual((out.width, out.height), (800, 1184))
        self.assertEqual([image.size for image in out.images], [(1536, 1024), (801, 1191)])
        kwargs = build_pipe_kwargs(out)
        self.assertEqual(kwargs["reference_resolution"], 0)
        self.assertEqual((kwargs["width"], kwargs["height"]), (800, 1184))

    def test_edit_explicit_canvas_does_not_resize_references(self):
        payload = {
            "type": "image_edit", "prompt": "combine", "width": 1024, "height": 1024,
            "output_resolution": 1024,
            "images": [_png_b64(1536, 1024), _png_b64(768, 512)],
        }
        out = normalize_job_input(payload, raw_keys=set(payload))
        kwargs = build_pipe_kwargs(out)
        self.assertEqual(kwargs["reference_resolution"], 0)
        self.assertEqual([image.size for image in kwargs["image"]], [(1536, 1024), (768, 512)])
        self.assertEqual((kwargs["width"], kwargs["height"]), (1024, 1024))

    def test_native_reference_outside_supported_size_is_rejected_before_inference(self):
        payload = {
            "type": "image_edit", "prompt": "combine", "width": 1024, "height": 1024,
            "images": [_png_b64(2816, 1024)],
        }
        with self.assertRaisesRegex(RequestError, "reference"):
            normalize_job_input(payload, raw_keys=set(payload))

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
        self.assertNotIn("sigmas", kwargs)
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
        self.assertEqual((kwargs["width"], kwargs["height"]), (256, 256))
        self.assertEqual(len(kwargs["image"]), 2)
        self.assertEqual(kwargs["negative_prompt"], "blur")
        self.assertEqual(kwargs["true_cfg_scale"], 2.5)
        self.assertEqual(kwargs["reference_resolution"], 0)
        self.assertEqual(kwargs["output_resolution"], 256)

    def test_schema_defaults_match_the_contract(self):
        steps = INPUT_SCHEMA["num_inference_steps"]
        self.assertEqual(steps["default"], 8)
        self.assertEqual(INPUT_SCHEMA["guidance_scale"]["default"], 1.0)
        self.assertEqual(INPUT_SCHEMA["width"]["default"], 1024)
        self.assertIsNone(INPUT_SCHEMA["output_resolution"]["default"])
        self.assertTrue(INPUT_SCHEMA["output_resolution"]["constraints"](None))
        self.assertFalse(INPUT_SCHEMA["width"]["constraints"](1000))
        self.assertFalse(steps["constraints"](0))
        self.assertTrue(steps["constraints"](40))
        self.assertTrue(steps["constraints"](6))
        self.assertTrue(steps["constraints"](8))
        self.assertFalse(INPUT_SCHEMA["num_images"]["constraints"](5))

    def test_omitted_steps_use_the_checkpoint_schedule(self):
        payload = {"prompt": "fox", "width": 1024, "height": 1024}
        out = normalize_job_input(payload, raw_keys=set(payload))
        self.assertEqual(out.num_inference_steps, 8)
        self.assertNotIn("sigmas", build_pipe_kwargs(out))

    def test_step_count_bounds_are_enforced_in_schema_and_normalizer(self):
        for steps in (0, -1, 201, True, 8.5, "8", None):
            with self.subTest(steps=steps):
                self.assertFalse(INPUT_SCHEMA["num_inference_steps"]["constraints"](steps))
                with self.assertRaisesRegex(RequestError, "num_inference_steps"):
                    _generate(num_inference_steps=steps)
        for steps in (1, 6, 25, 40, 100, 200):
            self.assertTrue(INPUT_SCHEMA["num_inference_steps"]["constraints"](steps))
            self.assertEqual(_generate(num_inference_steps=steps).num_inference_steps, steps)


if __name__ == "__main__":
    unittest.main()
