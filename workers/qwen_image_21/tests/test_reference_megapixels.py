"""Reference area limits through the real shared parser and image preprocessing."""

import base64
from io import BytesIO
from pathlib import Path
import sys
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
from qwen_image_21_infer.request import build_pipe_kwargs, normalize_job_input
from shared.api import QWEN_API, RequestError, parse_job_input


def image_data(width, height, *, mode="RGB"):
    buffer = BytesIO()
    color = (10, 20, 30, 80) if mode == "RGBA" else (10, 20, 30)
    Image.new(mode, (width, height), color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def request(images, **fields):
    raw = {"type": "image_edit", "prompt": "combine", "images": images, **fields}
    parsed = parse_job_input({"input": raw}, QWEN_API)
    return normalize_job_input(parsed.values, raw_keys=parsed.raw_keys)


class ReferenceMegapixelsTests(unittest.TestCase):
    def test_budget_is_shared_equally_while_output_keeps_the_full_area(self):
        source = image_data(2048, 2048)
        for variant in ("base", "turbo"):
            for count, expected_side, per_reference_pixels in (
                (1, 992, 1_000_000), (2, 704, 500_000),
                (3, 576, 333_333), (5, 416, 200_000), (10, 288, 100_000),
            ):
                with self.subTest(variant=variant, count=count):
                    norm = request([source] * count, model_options={"variant": variant})
                    self.assertEqual([img.size for img in norm.images], [(expected_side, expected_side)] * count)
                    self.assertTrue(all(img.width * img.height <= per_reference_pixels for img in norm.images))
                    self.assertLessEqual(sum(img.width * img.height for img in norm.images), 1_000_000)
                    self.assertEqual((norm.width, norm.height), (992, 992))
                    self.assertEqual(norm.reference_megapixels, 1.)

    def test_custom_total_budget_is_divided_without_reducing_output_area(self):
        norm = request([image_data(2048, 2048)] * 2, model_options={"reference_megapixels": 2.})
        self.assertEqual([img.size for img in norm.images], [(992, 992)] * 2)
        self.assertEqual((norm.width, norm.height), (1408, 1408))
        self.assertEqual(norm.reference_megapixels, 2.)

    def test_budget_below_the_combined_minimum_reference_area_is_rejected(self):
        source = image_data(32, 32)
        with self.assertRaises(RequestError):
            request([source] * 2, width=256, height=256,
                    model_options={"reference_megapixels": .001024})
        norm = request([source] * 2, width=256, height=256,
                       model_options={"reference_megapixels": .002048})
        self.assertEqual([img.size for img in norm.images], [(32, 32)] * 2)
        self.assertEqual(sum(img.width * img.height for img in norm.images), 2048)

    def test_small_reference_still_produces_a_megapixel_canvas_with_original_aspect(self):
        for variant in ("base", "turbo"):
            for source, target in (((100, 30), (1824, 544)), ((30, 100), (544, 1824)),
                                   ((100, 100), (992, 992))):
                with self.subTest(variant=variant, source=source):
                    norm = request([image_data(*source)], model_options={"variant": variant})
                    kwargs = build_pipe_kwargs(norm)
                    self.assertEqual((kwargs["width"], kwargs["height"]), target)
                    self.assertLessEqual(norm.width * norm.height, 1_000_000)
                    self.assertGreater(norm.width * norm.height, 950_000)
                    self.assertLessEqual(norm.images[0].width * norm.images[0].height, 10_000)

    def test_custom_megapixels_sets_automatic_canvas_even_for_small_last_reference(self):
        norm = request([image_data(400, 800), image_data(100, 30)],
                       model_options={"reference_megapixels": .5})
        self.assertEqual((norm.width, norm.height), (1280, 384))
        self.assertEqual(norm.images[-1].size, (96, 32))

    def test_explicit_output_resolution_overrides_megapixel_canvas_using_original_aspect(self):
        norm = request([image_data(100, 30)],
                       model_options={"reference_megapixels": .5, "output_resolution": 1024})
        self.assertEqual((norm.width, norm.height), (1856, 576))

    def test_shared_budget_caps_five_mixed_references_without_redistributing_unused_area(self):
        norm = request([image_data(w, h) for w, h in (
            (2048, 2048), (2048, 1024), (320, 320), (1001, 1001), (1024, 2048),
        )], width=1024, height=1024)
        kwargs = build_pipe_kwargs(norm)
        self.assertEqual([img.size for img in kwargs["image"]], [
            (416, 416), (608, 288), (320, 320), (416, 416), (288, 608),
        ])
        self.assertTrue(all(img.width * img.height <= 200_000 for img in kwargs["image"]))
        self.assertLessEqual(sum(img.width * img.height for img in kwargs["image"]), 1_000_000)
        self.assertEqual((kwargs["width"], kwargs["height"]), (1024, 1024))

    def test_custom_limit_preserves_alpha_and_explicit_output_canvas(self):
        norm = request([image_data(2048, 2048, mode="RGBA"), image_data(384, 256)],
                       width=2048, height=1024, model_options={"reference_megapixels": .5})
        self.assertEqual([img.size for img in norm.images], [(480, 480), (384, 256)])
        self.assertEqual(norm.images[0].mode, "RGBA")
        self.assertEqual(norm.images[0].getpixel((100, 100))[3], 80)
        self.assertEqual((norm.width, norm.height), (2048, 1024))

    def test_automatic_canvas_and_large_last_reference_follow_megapixel_budget(self):
        norm = request([image_data(512, 512), image_data(4000, 2000)])
        self.assertEqual((norm.width, norm.height), (1408, 704))
        self.assertEqual(norm.images[-1].size, (992, 480))

    def test_larger_requested_budget_is_used_without_upscaling(self):
        norm = request([image_data(2048, 2048)], model_options={"reference_megapixels": 2})
        self.assertEqual(norm.images[0].size, (1408, 1408))
        self.assertEqual((norm.width, norm.height), (1408, 1408))

    def test_rounding_stays_below_budget_and_existing_side_limit(self):
        norm = request([image_data(8192, 32), image_data(1024, 1024)],
                       width=1024, height=1024, model_options={"reference_megapixels": .1})
        self.assertEqual(norm.images[1].size, (192, 192))
        for img in norm.images:
            self.assertLessEqual(img.width * img.height, 50_000)
            self.assertLessEqual(max(img.size), 2752)
            self.assertEqual(img.width % 32, 0)
            self.assertEqual(img.height % 32, 0)

    def test_invalid_megapixels_are_client_errors(self):
        for value in (None, True, "1", 0, -1, .000001, 4.4, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(RequestError):
                parse_job_input({"input": {"prompt": "fox", "model_options": {
                    "reference_megapixels": value,
                }}}, QWEN_API)


if __name__ == "__main__":
    unittest.main()
