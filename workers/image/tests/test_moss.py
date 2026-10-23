"""MOSS participates in routing and owns no resources until selected."""

import importlib.util
from contextlib import nullcontext
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))


class MossIntegrationTests(unittest.TestCase):
    def test_first_turbo_load_starts_prefetch_before_pipeline_and_waits_for_its_exit(self):
        from image_worker import backends
        events = []
        from contextlib import contextmanager
        @contextmanager
        def prepare(directory, *, turbo=False, edit=False):
            self.assertTrue(turbo)
            self.assertFalse(edit)
            events.append('downloader_started')
            yield SimpleNamespace(wait=lambda component: events.append(component))
            events.append('downloader_closed')
        def load(directory, *, turbo=False, edit=False):
            self.assertTrue(turbo)
            self.assertFalse(edit)
            self.assertEqual(events, ['downloader_started'])
            events.append('pipeline_loaded')
            return object()
        with patch.object(backends, 'prefetch_moss_model', new=prepare), patch.object(
            backends, 'load_moss_pipeline', new=load
        ):
            backend = backends.MossBackend()
            backend.load(SimpleNamespace(model_variant='turbo', type='image_generate'))
        self.assertEqual(events, ['downloader_started', 'pipeline_loaded', 'all', 'downloader_closed'])

    def test_warm_base_to_turbo_prepares_adapter_before_inference_and_propagates_download_failure(self):
        from image_worker import backends
        events = []
        from contextlib import contextmanager
        failed = False
        @contextmanager
        def prepare(directory, *, turbo, lora_only):
            self.assertTrue(turbo and lora_only)
            def wait(component):
                if failed:
                    raise RuntimeError('adapter download failed')
                events.append(component)
            yield SimpleNamespace(wait=wait)
        runtime = SimpleNamespace(load_turbo=lambda: events.append('adapter_loaded'), close=lambda: events.append('closed'),
                                  run=lambda *args: events.append('inference'))
        with patch.object(backends, 'prefetch_moss_model', new=prepare):
            backend = backends.MossBackend()
            backend.runtime = runtime
            norm = SimpleNamespace(model_variant='turbo', type='image_generate')
            backend.infer(norm, 0, [])
            self.assertEqual(events, ['adapter_loaded', 'all', 'inference'])
            events.clear()
            failed = True
            with self.assertRaisesRegex(RuntimeError, 'adapter download failed'):
                backend.infer(norm, 0, [])
            self.assertEqual(events, ['adapter_loaded', 'closed'])
            self.assertIsNone(backend.runtime)

    def test_edit_prefetches_selected_adapter_on_cold_and_warm_requests(self):
        from image_worker import backends
        from contextlib import contextmanager
        events = []
        @contextmanager
        def prepare(directory, *, turbo=False, edit=False, lora_only=False):
            self.assertFalse(turbo)
            self.assertTrue(edit)
            events.append('warm_download' if lora_only else 'cold_download')
            yield SimpleNamespace(wait=lambda component: events.append(component))
            events.append('downloader_exit')
        runtime = SimpleNamespace(load_edit=lambda: events.append('adapter_loaded'),
                                  run=lambda *args: events.append('inference'))
        def load(directory, *, turbo=False, edit=False):
            self.assertFalse(turbo)
            self.assertTrue(edit)
            self.assertEqual(events, ['cold_download'])
            return runtime
        with patch.object(backends, 'prefetch_moss_model', new=prepare), patch.object(
            backends, 'load_moss_pipeline', new=load
        ):
            backend = backends.MossBackend()
            norm = SimpleNamespace(model_variant='base', type='image_edit')
            backend.load(norm)
            backend.infer(norm, 0, [])
        self.assertEqual(events, ['cold_download', 'all', 'downloader_exit', 'warm_download',
                                  'adapter_loaded', 'all', 'downloader_exit', 'inference'])

    def test_router_resolves_variants_and_requires_edit_references_before_loading(self):
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
        self.assertIn('image_edit requires', result['error'])
        self.assertEqual(calls, [])

    def test_router_accepts_edit_with_decoded_references_and_preserves_seed_zero(self):
        from image_worker.backends import REGISTRY
        from image_worker.api import generate_image
        from shared.api import GenerationResult
        from shared.api.images import images_to_base64_data_urls
        class Manager:
            registry = REGISTRY
            def close(manager):
                raise AssertionError('Valid edit must not close the manager')
            def infer(manager, model, norm, seed, loras):
                self.assertEqual((model, seed, norm.model_variant), ('moss_image', 0, 'base'))
                self.assertEqual(norm.images[0].getpixel((0, 0)), (255, 0, 0))
                return GenerationResult([Image.new('RGB', (256, 256))], [], {'variant': norm.model_variant})
        result = generate_image({'input': {'model': 'moss_image', 'type': 'image_edit', 'prompt': 'Make blue',
                                           'seed': 0, 'images': images_to_base64_data_urls([Image.new('RGB', (64, 64), 'red')])}},
                                Manager(), deliver=lambda *args: ['image'])
        self.assertNotIn('error', result)
        self.assertEqual((result['type'], result['num_refs'], result['num_inference_steps']), ('image_edit', 1, 30))

    def test_cli_sends_edit_references_and_defaults_to_sft(self):
        import tempfile
        scripts = Path(__file__).parents[3] / 'scripts'
        spec = importlib.util.spec_from_file_location('_moss_edit_client', scripts / 'moss_image.py')
        client = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(client)
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'ref.png'
            Image.new('RGB', (32, 32), 'red').save(image)
            args = client._build_parser().parse_args(['-i', str(image), '--prompt', 'make blue', '--seed', '0'])
            with patch.object(client, '_submit_job', return_value='job') as submit, patch.object(
                client, '_poll_job', return_value={}
            ), patch.object(client, '_save_result', return_value=0):
                self.assertEqual(client._run_generate(args, endpoint_id='common', api_key='key'), 0)
            payload = submit.call_args.args[2]
            self.assertEqual((payload['type'], payload['model_options'], payload['seed']),
                             ('image_edit', {'variant': 'base'}, 0))
            self.assertTrue(payload['images'][0].startswith('data:image/png;base64,'))

    def test_cli_rejects_turbo_batch_and_excess_references_without_submitting(self):
        scripts = Path(__file__).parents[3] / 'scripts'
        spec = importlib.util.spec_from_file_location('_moss_edit_invalid_client', scripts / 'moss_image.py')
        client = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(client)
        for flags in (['--variant', 'turbo'], ['--num-images', '2'], ['-i', 'other.png'] * 4):
            args = client._build_parser().parse_args(['-i', 'ref.png', '--prompt', 'make blue', *flags])
            with self.subTest(flags=flags), patch.object(client, '_submit_job', side_effect=AssertionError('Invalid edit submitted')):
                self.assertEqual(client._run_generate(args, endpoint_id='common', api_key='key'), 1)

    def test_backend_defers_download_and_closes_runtime(self):
        from image_worker import backends
        backend = backends.REGISTRY['moss_image']()
        self.assertIsNone(backend.runtime)
        class Runtime:
            closed = False
            def close(self):
                self.closed = True
        runtime = Runtime()
        with patch.object(backends, 'prefetch_moss_model', return_value=nullcontext(None)) as prefetch, patch.object(
            backends, 'load_moss_pipeline', return_value=runtime
        ), patch.dict(os.environ, {'MOSS_MODEL_DIR': '/models/selected-moss'}):
            backend.load()
            self.assertEqual(prefetch.call_args.args[0], Path('/models/selected-moss'))
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
