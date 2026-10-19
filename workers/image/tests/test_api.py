"""Routing preserves each model's existing request contract."""

import base64
import copy
from io import BytesIO
from pathlib import Path
import sys
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))


class RoutingTests(unittest.TestCase):
    def setUp(self):
        try:
            from image_worker.api import generate_image
        except ImportError:
            self.fail('The common image request dispatcher is missing')
        from image_worker.backends import REGISTRY
        from shared.api import GenerationResult
        self.generate = generate_image
        self.calls = []
        calls = self.calls

        class Manager:
            registry = REGISTRY
            closed = False

            def infer(self, model, norm, seed, raw_loras):
                calls.append((model, norm, seed, raw_loras))
                return GenerationResult([Image.new('RGB', (norm.width, norm.height))], [], {})

            def close(self):
                self.closed = True

        self.manager = Manager()

    def run_job(self, values):
        return self.generate({'id': 'test', 'input': values}, self.manager,
                             deliver=lambda images, job: ['data:image/png;base64,test'])

    def test_default_krea_and_explicit_qwen_apply_distinct_defaults_and_keep_seed_zero(self):
        for model, options, steps, guidance in (
            (None, {}, 8, 0.), ('krea2', {}, 8, 0.),
            ('qwen_image_21', {}, 8, 1.), ('qwen_image_21', {'variant': 'base'}, 40, 1.),
        ):
            values = {'prompt': 'fox', 'seed': 0, 'model_options': options}
            if model is not None:
                values['model'] = model
            before = copy.deepcopy(values)
            output = self.run_job(values)
            self.assertNotIn('error', output)
            self.assertEqual(output['model'], model or 'krea2')
            self.assertEqual(output['num_inference_steps'], steps)
            self.assertEqual(output['guidance_scale'], guidance)
            self.assertEqual(output['seed'], 0)
            self.assertEqual(self.calls[-1][0], model or 'krea2')
            if options:
                self.assertEqual(self.calls[-1][1].model_variant, 'base')
            self.assertEqual(values, before, 'Routing must not mutate the caller input')

    def test_bad_models_do_not_reach_or_close_the_manager(self):
        for model in ('unknown', '', None, 7, [], {}):
            with self.subTest(model=model):
                output = self.run_job({'model': model, 'prompt': 'fox'})
                self.assertIn('error', output)
                self.assertNotIn('refresh_worker', output)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.manager.closed)

    def test_unknown_fields_and_other_models_options_stay_invalid(self):
        for values in (
            {'model': 'krea2', 'invented': True},
            {'model': 'krea2', 'model_options': {'variant': 'base'}},
            {'model': 'qwen_image_21', 'model_options': {'mu': 1.15}},
            {'model': 'krea2', 'sampler': 'heun'},
        ):
            output = self.run_job({'prompt': 'fox', **values})
            self.assertIn('error', output)
        self.assertEqual(self.calls, [])

    def test_reference_limits_are_selected_before_inference(self):
        buffer = BytesIO()
        Image.new('RGB', (64, 96)).save(buffer, format='PNG')
        source = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
        values = {'type': 'image_edit', 'prompt': 'edit', 'images': [source] * 3}
        self.assertIn('error', self.run_job({'model': 'krea2', **values}))
        self.assertEqual(self.calls, [])
        output = self.run_job({'model': 'qwen_image_21', **values})
        self.assertNotIn('error', output)
        self.assertEqual(output['num_refs'], 3)
        self.assertEqual(self.calls[-1][0], 'qwen_image_21')

    def test_invalid_envelopes_return_client_errors(self):
        for job in (None, [], {}, {'input': None}):
            output = self.generate(job, self.manager)
            self.assertIn('error', output)
        self.assertEqual(self.calls, [])

    def test_cuda_oom_releases_the_active_backend_and_requests_worker_refresh(self):
        import torch

        def fail(*args):
            raise torch.cuda.OutOfMemoryError('test allocation')

        self.manager.infer = fail
        output = self.run_job({'model': 'qwen_image_21', 'prompt': 'fox'})
        self.assertIn('error', output)
        self.assertTrue(output['refresh_worker'])
        self.assertTrue(self.manager.closed)
