"""MOSS generation/edit settings and invalid requests resolve before loading."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).parents[2]), str(Path(__file__).parents[1])]


class RequestTests(unittest.TestCase):
    def image(self, size=(768, 512)):
        from PIL import Image
        from shared.api.images import images_to_base64_data_urls
        return images_to_base64_data_urls([Image.new('RGB', size, 'red')])[0]

    def test_edit_defaults_select_sft_without_reusing_generation_capacity(self):
        with patch('shared.megapixels.detect_vram_gib', return_value=47_665_709_056 / 1024**3):
            norm = self.normalize(type='image_edit', images=[self.image()])
        self.assertEqual((norm.model_variant, norm.num_inference_steps, norm.guidance_scale), ('base', 30, 4.))
        self.assertEqual(norm.reference_megapixels, .262144)
        self.assertEqual(norm.images[0].size, (768, 512))
        self.assertEqual((norm.width, norm.height), (1024, 1024))
        self.assertEqual(norm.uncond_skip_layers, (22,))

    def test_edit_accepts_four_references_and_explicit_settings(self):
        norm = self.normalize(type='image_edit', images=[self.image()] * 4,
                              model_options={'variant': 'base', 'uncond_skip_layers': []},
                              num_inference_steps=12, guidance_scale=2., reference_megapixels=.5,
                              width=768, height=512)
        self.assertEqual(len(norm.images), 4)
        self.assertEqual((norm.num_inference_steps, norm.guidance_scale, norm.reference_megapixels), (12, 2., .5))
        self.assertEqual(norm.uncond_skip_layers, ())

    def test_edit_rejects_turbo_and_invalid_references_before_loading(self):
        from shared.api import RequestError
        image = self.image()
        for values in (
            {'images': []}, {'images': [image] * 5}, {'images': ['invalid']},
            {'images': [image], 'num_images': 2},
            {'images': [image], 'model_options': {'variant': 'turbo'}},
        ):
            with self.subTest(values={k: v for k, v in values.items() if k != 'images'}), self.assertRaises(RequestError):
                self.normalize(type='image_edit', **values)

    def normalize(self, **values):
        from moss_image_infer.request import normalize_job_input
        from shared.api import MOSS_API, parse_job_input
        parsed = parse_job_input({'input': {'prompt': 'fox', **values}}, MOSS_API)
        return normalize_job_input(parsed.values, raw_keys=parsed.raw_keys)

    def test_max_budget_uses_calibrated_moss_profiles_on_measured_gpu_tiers(self):
        for total_bytes in (25_250_627_584, 47_665_709_056):
            for variant in ('base', 'turbo'):
                with self.subTest(total_bytes=total_bytes, variant=variant), patch(
                    'shared.megapixels.detect_vram_gib', return_value=total_bytes / 1024**3
                ):
                    norm = self.normalize(model_options={'variant': variant}, width=2048, height=2048)
                    self.assertEqual(norm.requested_reference_megapixels, 'max')
                    self.assertEqual(norm.reference_megapixels, 2048 * 2048 / 1_000_000)
                    self.assertEqual(norm.vram_gib, total_bytes / 1024**3)
                    self.assertEqual((norm.width, norm.height), (2048, 2048))

    def test_uncalibrated_tiers_keep_fallback_budget_and_explicit_canvas(self):
        for vram_gib in (None, 15., 31., 60.):
            for variant in ('base', 'turbo'):
                with self.subTest(vram_gib=vram_gib, variant=variant), patch(
                    'shared.megapixels.detect_vram_gib', return_value=vram_gib
                ):
                    norm = self.normalize(model_options={'variant': variant}, width=768, height=1024)
                    self.assertEqual(norm.reference_megapixels, 1.)
                    self.assertEqual((norm.width, norm.height), (768, 1024))

    def test_variant_defaults_and_explicit_values(self):
        for variant, steps, cfg, skip in (('base', 50, 3., (22,)), ('turbo', 8, 1., ())):
            with self.subTest(variant=variant):
                norm = self.normalize(model_options={'variant': variant})
                self.assertEqual((norm.num_inference_steps, norm.guidance_scale, norm.uncond_skip_layers),
                                 (steps, cfg, skip))
                self.assertEqual((norm.width, norm.height, norm.shift), (1024, 1024, 5.))
        norm = self.normalize(seed=0, num_inference_steps=4, width=768, height=1024)
        self.assertEqual((norm.seed, norm.num_inference_steps, norm.model_variant), (0, 4, 'turbo'))
        norm = self.normalize(guidance_scale=4., num_inference_steps=60,
                              model_options={'variant': 'base', 'shift': 4., 'uncond_skip_layers': []})
        self.assertEqual((norm.guidance_scale, norm.shift, norm.uncond_skip_layers), (4., 4., ()))

    def test_rejects_edits_and_incompatible_turbo_settings(self):
        from shared.api import RequestError
        for values in (
            {'type': 'image_edit'}, {'images': ['not decoded']}, {'width': 513},
            {'guidance_scale': 3.}, {'negative_prompt': 'bad anatomy'},
            {'model_options': {'uncond_skip_layers': [22]}},
            {'model_options': {'variant': 'other'}}, {'sampler': 'heun'},
            {'model_options': {'shift': 0.}}, {'loras': [{'name': 'unknown'}]},
            {'model_options': {'variant': 'base', 'uncond_skip_layers': [42]}},
            {'model_options': {'variant': 'base', 'uncond_skip_layers': [True]}},
            {'model_options': {'variant': 'base', 'uncond_skip_layers': [22, 22]}},
        ):
            with self.subTest(values=values), self.assertRaises(RequestError):
                self.normalize(**values)

    def test_json_prompt_is_preserved_and_size_is_explicit(self):
        prompt = '{"summary":{"caption":"Привет"},"items":[]}'
        norm = self.normalize(prompt=prompt, width=2048, height=1024)
        self.assertEqual(norm.prompt, prompt)
        self.assertEqual((norm.width, norm.height), (2048, 1024))


if __name__ == '__main__':
    unittest.main()
