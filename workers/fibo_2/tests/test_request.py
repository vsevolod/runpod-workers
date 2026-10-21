"""Fibo's public base/turbo modes and native structured-caption boundaries."""

import base64
from io import BytesIO
from pathlib import Path
import sys
import unittest

from PIL import Image

WORKERS = Path(__file__).parents[2]
sys.path[:0] = [str(WORKERS), str(WORKERS / 'fibo_2')]

from shared.api import FIBO_API, RequestError, parse_job_input


def reference():
    buf = BytesIO()
    Image.new('RGB', (48, 32), 'red').save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


class FiboRequestTests(unittest.TestCase):
    def normalize(self, **fields):
        try:
            from fibo_2_infer.request import normalize_job_input
        except ModuleNotFoundError:
            self.fail('Fibo needs a request normalizer before loading its model')
        parsed = parse_job_input({'input': {'prompt': {'short_description': 'A red fox.'}, **fields}}, FIBO_API)
        return normalize_job_input(parsed.values, raw_keys=parsed.raw_keys)

    def test_base_generation_and_edit_defaults_keep_native_auto_edit_dimensions(self):
        request = self.normalize(seed=0)
        self.assertEqual((request.model_variant, request.num_inference_steps, request.guidance_scale,
                          request.width, request.height, request.seed), ('base', 30, 5., 1024, 1024, 0))
        edit = self.normalize(type='image_edit', images=[reference()],
                              prompt={'edit_instruction': 'Make the fox blue.'})
        self.assertEqual((edit.model_variant, len(edit.images)), ('base', 1))
        self.assertIsNone(edit.width)
        self.assertIsNone(edit.height)

    def test_turbo_edits_use_four_steps_without_cfg(self):
        request = self.normalize(type='image_edit', images=[reference()],
                                 prompt={'edit_instruction': 'Make it blue.'},
                                 model_options={'variant': 'turbo'})
        self.assertEqual((request.model_variant, request.num_inference_steps, request.guidance_scale),
                          ('turbo', 4, 1.))
        with self.assertRaisesRegex(RequestError, 'turbo.*image_edit'):
            self.normalize(model_options={'variant': 'turbo'})

    def test_turbo_rejects_ineffective_cfg_negative_prompt_and_user_loras(self):
        for extra in ({'guidance_scale': 2.}, {'negative_prompt': {'short_description': 'blur'}},
                      {'loras': [{'name': 'adapter', 'strength': 1.}]}):
            with self.subTest(extra=extra), self.assertRaises(RequestError):
                self.normalize(type='image_edit', images=[reference()],
                               prompt={'edit_instruction': 'Make it blue.'},
                               model_options={'variant': 'turbo'}, **extra)

    def test_json_shapes_are_validated_before_model_loading(self):
        for prompt in ({'short_description': ' '}, {'short_description': 42},
                       {'short_description': 'fox', 'edit_instruction': 'blue'},
                       {'short_description': 'fox', 'objects': [float('nan')]},
                       {'structured_caption': 'fox'}, {'objects': []}):
            with self.subTest(prompt=prompt), self.assertRaises(RequestError):
                self.normalize(prompt=prompt)
        for prompt in ({'short_description': 'fox'}, {'edit_instruction': ''},
                       {'edit_instruction': 42},
                       {'edit_instruction': 'Blue.', 'structured_caption': {'short_description': 'fox'},
                        'unexpected_sibling': True}):
            with self.subTest(prompt=prompt), self.assertRaises(RequestError):
                self.normalize(type='image_edit', images=[reference()], prompt=prompt)

    def test_nested_caption_and_four_ordered_references_are_supported(self):
        prompt = {'structured_caption': {'short_description': 'A blue fox.'},
                  'edit_instruction': 'Use the color from <image_4>.'}
        request = self.normalize(type='image_edit', images=[reference()] * 4, prompt=prompt,
                                 width=1536, height=1024, guidance_scale=3., num_inference_steps=20)
        self.assertEqual(request.prompt, prompt)
        self.assertEqual((len(request.images), request.width, request.height), (4, 1536, 1024))
        self.assertEqual((request.num_inference_steps, request.guidance_scale), (20, 3.))
        with self.assertRaises(RequestError):
            self.normalize(type='image_edit', images=[reference()] * 5, prompt=prompt)

    def test_reference_budget_is_validated_before_loading(self):
        edit = dict(type='image_edit', images=[reference()], prompt={'edit_instruction': 'Blue.'})
        request = self.normalize(**edit, width=1024, height=1024)
        self.assertEqual(request.reference_megapixels, 1.048576)
        for fields in ({'width': 1024, 'height': 1024, 'reference_megapixels': .5},
                       {'reference_megapixels': .0001}):
            with self.subTest(fields=fields), self.assertRaises(RequestError):
                self.normalize(**edit, **fields)

    def test_partial_dimensions_and_oversized_canvases_are_rejected(self):
        for dimensions in ({'width': 1024}, {'height': 1024}, {'width': 4096, 'height': 4096}):
            with self.subTest(dimensions=dimensions), self.assertRaises(RequestError):
                self.normalize(**dimensions)


if __name__ == '__main__':
    unittest.main()
