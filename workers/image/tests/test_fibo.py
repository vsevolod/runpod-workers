"""Fibo routing, selected checkpoints, failure cleanup and the JSON client."""
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
sys.path.insert(0, str(Path(__file__).parents[1]))


class FiboIntegrationTests(unittest.TestCase):
    def test_variant_switch_closes_before_download_and_reuses_resident_variant(self):
        from image_worker import backends
        events = []
        @contextmanager
        def prepare(directory, *, variant):
            events.append(('download', variant, directory))
            yield SimpleNamespace(wait=lambda component: events.append(('wait', component)))
        class Runtime:
            def __init__(self, variant): self.variant = variant
            def close(self): events.append(('close', self.variant))
            def run(self, norm, seed): return (self.variant, seed)
        def load(directory, *, variant):
            events.append(('load', variant))
            return Runtime(variant)
        with patch.object(backends, 'prefetch_fibo_model', new=prepare), patch.object(
            backends, 'load_fibo_pipeline', new=load
        ), patch.dict(os.environ, {'FIBO_MODEL_DIR': '/models/selected-fibo'}):
            backend = backends.REGISTRY['fibo_2']()
            self.assertEqual(events, [])
            base, turbo = (SimpleNamespace(model_variant=v) for v in ('base', 'turbo'))
            backend.load(base)
            self.assertEqual(backend.infer(base, 0, []), ('base', 0))
            self.assertEqual(backend.infer(turbo, 1, []), ('turbo', 1))
            backend.close()
            backend.close()
        self.assertEqual(events, [('download', 'base', Path('/models/selected-fibo/base')),
            ('load', 'base'), ('wait', 'all'), ('close', 'base'),
            ('download', 'turbo', Path('/models/selected-fibo/turbo')),
            ('load', 'turbo'), ('wait', 'all'), ('close', 'turbo')])

    def test_late_downloader_failure_closes_loaded_runtime(self):
        from image_worker import backends
        from unittest.mock import Mock
        runtime = SimpleNamespace(close=Mock())
        @contextmanager
        def prepare(*args, **kwargs):
            yield SimpleNamespace(wait=lambda component: (_ for _ in ()).throw(RuntimeError('late download failure')))
        with patch.object(backends, 'prefetch_fibo_model', new=prepare), patch.object(
            backends, 'load_fibo_pipeline', return_value=runtime
        ):
            backend = backends.FiboBackend()
            with self.assertRaisesRegex(RuntimeError, 'late download failure'):
                backend.load(SimpleNamespace(model_variant='base'))
            self.assertIsNone(backend.runtime)
        runtime.close.assert_called_once()

    def test_router_accepts_json_and_rejects_turbo_generation_before_loading(self):
        from image_worker.backends import REGISTRY
        from image_worker.api import generate_image
        from shared.api import GenerationResult
        calls = []
        class Manager:
            registry = REGISTRY
            def close(self): raise AssertionError('Unexpected manager cleanup')
            def infer(self, model, norm, seed, loras):
                calls.append((model, norm, seed))
                return GenerationResult([Image.new('RGB', (32, 32))], [], {'variant': norm.model_variant})
        result = generate_image({'input': {'model': 'fibo_2', 'prompt': {'short_description': 'fox'}, 'seed': 0}},
                                Manager(), deliver=lambda *args: ['image'])
        self.assertNotIn('error', result)
        self.assertEqual((result['model'], result['seed'], result['num_inference_steps']), ('fibo_2', 0, 30))
        self.assertEqual(calls[0][1].prompt, {'short_description': 'fox'})
        result = generate_image({'input': {'model': 'fibo_2', 'prompt': {'short_description': 'fox'},
                                           'model_options': {'variant': 'turbo'}}}, Manager())
        self.assertIn('only image_edit', result['error'])
        self.assertEqual(len(calls), 1)
