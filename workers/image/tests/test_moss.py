"""MOSS participates in routing and owns no resources until selected."""

import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))


class MossIntegrationTests(unittest.TestCase):
    def test_router_resolves_variant_defaults_and_rejects_edits_before_loading(self):
        from image_worker.backends import REGISTRY
        from image_worker.api import generate_image
        from shared.api import GenerationResult
        calls = []
        class Manager:
            registry = REGISTRY
            def close(self):
                raise AssertionError('Valid generation and rejected edits must not close the manager')
            def infer(self, model, norm, seed, loras):
                calls.append((model, norm, seed))
                return GenerationResult([Image.new('RGB', (norm.width, norm.height))], [], {})
        for variant, steps, cfg in (('base', 50, 3.), ('turbo', 8, 1.)):
            result = generate_image({'input': {'model': 'moss_image', 'prompt': 'fox', 'seed': 0,
                                              'model_options': {'variant': variant}}}, Manager(),
                                    deliver=lambda *args: ['image'])
            self.assertNotIn('error', result)
            self.assertEqual((result['model'], result['num_inference_steps'], result['guidance_scale'], result['seed']),
                             ('moss_image', steps, cfg, 0))
        calls.clear()
        result = generate_image({'input': {'model': 'moss_image', 'prompt': 'fox', 'type': 'image_edit'}}, Manager())
        self.assertIn('only image_generate', result['error'])
        self.assertEqual(calls, [])

    def test_backend_defers_download_and_closes_runtime(self):
        from image_worker import backends
        backend = backends.REGISTRY['moss_image']()
        self.assertIsNone(backend.runtime)
        class Runtime:
            closed = False
            def close(self):
                self.closed = True
        runtime = Runtime()
        with patch.object(backends, 'PrefetchSession') as prefetch, patch.object(
            backends, 'load_moss_pipeline', return_value=runtime
        ), patch.dict(os.environ, {'MOSS_MODEL_DIR': '/models/selected-moss'}):
            backend.load()
            command = prefetch.call_args.args[0]
            self.assertEqual(command[command.index('--output') + 1], '/models/selected-moss')
            self.assertIs(backend.runtime, runtime)
        backend.close()
        backend.close()
        self.assertTrue(runtime.closed)
        self.assertIsNone(backend.runtime)

    def test_cli_keeps_server_defaults_and_can_override_steps_cfg_and_seed_zero(self):
        scripts = Path(__file__).parents[3] / 'scripts'
        spec = importlib.util.spec_from_file_location('_moss_client', scripts / 'moss_image.py')
        client = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(client)
        for variant in ('base', 'turbo'):
            args = client._build_parser().parse_args(['--prompt', 'fox', '--variant', variant, '--seed', '0'])
            with patch.object(client, '_submit_job', return_value='job') as submit, patch.object(
                client, '_poll_job', return_value={}
            ), patch.object(client, '_save_result', return_value=0):
                self.assertEqual(client._run_generate(args, endpoint_id='common', api_key='key'), 0)
            payload = submit.call_args.args[2]
            self.assertEqual(payload['model'], 'moss_image')
            self.assertEqual(payload['model_options'], {'variant': variant})
            self.assertEqual(payload['seed'], 0)
            self.assertNotIn('num_inference_steps', payload)
            self.assertNotIn('guidance_scale', payload)
        args = client._build_parser().parse_args(['--prompt', 'fox', '--variant', 'base', '--steps', '30', '--cfg', '4'])
        with patch.object(client, '_submit_job', return_value='job') as submit, patch.object(
            client, '_poll_job', return_value={}
        ), patch.object(client, '_save_result', return_value=0):
            client._run_generate(args, endpoint_id='common', api_key='key')
        self.assertEqual(submit.call_args.args[2]['num_inference_steps'], 30)
        self.assertEqual(submit.call_args.args[2]['guidance_scale'], 4.)
