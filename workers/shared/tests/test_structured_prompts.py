"""Structured prompts belong to Fibo; other models retain their text contract."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[2]))

from shared import api


class StructuredPromptTests(unittest.TestCase):
    def profile(self):
        profile = getattr(api, 'FIBO_API', None)
        self.assertIsNotNone(profile, 'The shared API must expose the Fibo model profile')
        return profile

    def test_fibo_preserves_json_and_uses_base_generation_defaults(self):
        scene = {'short_description': 'A red fox in snow.', 'text_render': []}
        raw = {'input': {'prompt': scene, 'seed': 0}}
        original = deepcopy(raw)
        profile = self.profile()
        parsed = api.parse_job_input(raw, profile)
        request = api.normalize_common_fields(parsed.values, profile, raw_keys=parsed.raw_keys)
        self.assertEqual((request.prompt, request.seed), (scene, 0))
        self.assertEqual((request.num_inference_steps, request.guidance_scale), (30, 5.))
        self.assertIsNone(request.negative_prompt)
        request.prompt['text_render'].append({'text': 'FOX'})
        self.assertEqual(raw, original)

    def test_fibo_negative_vgl_is_preserved_as_an_object(self):
        profile = self.profile()
        negative = {'short_description': 'Blurry, monochrome scene.'}
        parsed = api.parse_job_input({'input': {
            'prompt': {'short_description': 'A red fox.'}, 'negative_prompt': negative,
        }}, profile)
        request = api.normalize_common_fields(parsed.values, profile, raw_keys=parsed.raw_keys)
        self.assertEqual(request.negative_prompt, negative)

    def test_fibo_rejects_string_and_empty_prompt_objects(self):
        profile = self.profile()
        for prompt in ('{"short_description":"fox"}', {}, [], None):
            with self.subTest(prompt=prompt), self.assertRaises(api.RequestError):
                parsed = api.parse_job_input({'input': {'prompt': prompt}}, profile)
                api.normalize_common_fields(parsed.values, profile, raw_keys=parsed.raw_keys)

    def test_existing_models_still_require_text_and_keep_their_cfg_defaults(self):
        for profile in (api.KREA_API, api.QWEN_API, api.MOSS_API):
            with self.subTest(model=profile.name):
                with self.assertRaises(api.RequestError):
                    api.parse_job_input({'input': {'prompt': {'short_description': 'fox'}}}, profile)
                parsed = api.parse_job_input({'input': {'prompt': '  fox  '}}, profile)
                request = api.normalize_common_fields(parsed.values, profile, raw_keys=parsed.raw_keys)
                self.assertEqual(request.prompt, 'fox')


if __name__ == '__main__':
    unittest.main()
