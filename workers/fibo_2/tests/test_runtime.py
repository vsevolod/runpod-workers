"""The runtime serializes VGL at the model boundary and frees inference hooks."""

import importlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from PIL import Image

sys.path[:0] = [str(Path(__file__).parents[2]), str(Path(__file__).parents[1])]


class FiboRuntimeTests(unittest.TestCase):
    def runtime_module(self):
        try:
            return importlib.import_module('fibo_2_infer.pipeline')
        except ModuleNotFoundError:
            self.fail('Fibo needs a runtime adapter for generation and editing')

    def request(self, *, edit=False):
        return SimpleNamespace(
            prompt={'short_description': 'A blue fox.', **({'edit_instruction': 'Make it blue.'} if edit else {})},
            negative_prompt=None, type='image_edit' if edit else 'image_generate',
            model_variant='turbo' if edit else 'base', width=512, height=512,
            reference_megapixels=1.048576,
            images=(Image.new('RGB', (512, 512), 'red'),) if edit else (),
            num_inference_steps=4 if edit else 30, guidance_scale=1. if edit else 5., num_images=1,
        )

    def test_generation_passes_native_null_negative_and_serialized_vgl(self):
        module = self.runtime_module()
        class Model:
            hooks_freed = False
            image_processor = SimpleNamespace()
            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(images=[Image.new('RGB', (512, 512), 'blue')])
            def maybe_free_model_hooks(self):
                self.hooks_freed = True
            def remove_all_hooks(self):
                pass
        pipe = Model()
        with TemporaryDirectory() as root:
            runtime = module.Runtime(pipe, root, 'base')
            result = runtime.run(self.request(), 0)
        self.assertEqual(result.images[0].size, (512, 512))
        self.assertEqual(json.loads(pipe.kwargs['prompt']), {'short_description': 'A blue fox.'})
        self.assertIsNone(pipe.kwargs['negative_prompt'])
        self.assertIsNone(pipe.kwargs['image'])
        self.assertEqual(pipe.kwargs['generator'].initial_seed(), 0)
        self.assertEqual((pipe.kwargs['num_inference_steps'], pipe.kwargs['guidance_scale']), (30, 5.))
        self.assertTrue(pipe.hooks_freed)
        self.assertEqual(result.model_options, {'variant': 'base'})
        self.assertEqual(result.metadata['model_id'], 'briaai/Fibo-2.0-dev')

    def test_edit_keeps_reference_order_and_releases_hooks_on_failure(self):
        module = self.runtime_module()
        class Model:
            freed = False
            image_processor = SimpleNamespace()
            def __call__(self, **kwargs):
                self.kwargs = kwargs
                raise RuntimeError('denoising failed')
            def maybe_free_model_hooks(self):
                self.freed = True
            def remove_all_hooks(self):
                pass
        pipe = Model()
        request = self.request(edit=True)
        with TemporaryDirectory() as root:
            runtime = module.Runtime(pipe, root, 'turbo')
            with self.assertRaisesRegex(RuntimeError, 'denoising failed'):
                runtime.run(request, 4)
        self.assertEqual(pipe.kwargs['image'], list(request.images))
        self.assertEqual(json.loads(pipe.kwargs['prompt'])['edit_instruction'], 'Make it blue.')
        self.assertEqual((pipe.kwargs['num_inference_steps'], pipe.kwargs['guidance_scale']), (4, 1.))
        self.assertTrue(pipe.freed)

    def test_native_reference_preprocessing_respects_the_combined_budget(self):
        module = self.runtime_module()
        processor = module.BudgetedImageProcessor(vae_scale_factor=16)
        processor.reference_megapixels = .01
        images = [Image.new('RGB', (64, 128)), Image.new('RGB', (128, 64))]
        tensors, encoder_images = processor.preprocess(images, height=512, width=512)
        self.assertEqual([tuple(t.shape[-2:]) for t in tensors], [(96, 48), (48, 96)])
        self.assertEqual([image.size for image in encoder_images], [(48, 96), (96, 48)])
        self.assertLessEqual(sum(t.shape[-1] * t.shape[-2] for t in tensors), 10000)


if __name__ == '__main__':
    unittest.main()
