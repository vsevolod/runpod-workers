"""Pinned component readiness and interrupted-transfer repair with real local files."""

import importlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).parents[2]), str(Path(__file__).parents[1])]


def files():
    result = {name: b'{}' for name in (
        'transformer/config.json', 'text_encoder/config.json', 'text_encoder/generation_config.json',
        'processor/processor_config.json', 'processor/tokenizer_config.json', 'processor/tokenizer.json',
        'vae/config.json', 'scheduler/scheduler_config.json')}
    result.update({
        'model_index.json': b'{"_class_name":"BriaFibo2Pipeline"}',
        'transformer/diffusion_pytorch_model.safetensors.index.json': json.dumps({'weight_map': {
            'first': 'diffusion_pytorch_model-00001-of-00002.safetensors',
            'last': 'diffusion_pytorch_model-00002-of-00002.safetensors'}}).encode(),
        'transformer/diffusion_pytorch_model-00001-of-00002.safetensors': b'first transformer shard',
        'transformer/diffusion_pytorch_model-00002-of-00002.safetensors': b'second transformer shard',
        'text_encoder/model.safetensors': b'text encoder',
        'processor/chat_template.jinja': b'chat template',
        'vae/diffusion_pytorch_model.safetensors': b'vae weights',
        'fonts/Aileron-Regular.otf': b'font-a', 'fonts/DejaVuSans.ttf': b'font-b',
        'fonts/LICENSE-Aileron.txt': b'font-a license', 'fonts/LICENSE-DejaVu.txt': b'font-b license',
    })
    return result


class FiboStoreTests(unittest.TestCase):
    def store(self):
        try:
            return importlib.import_module('fibo_2_infer.model_store')
        except ModuleNotFoundError:
            self.fail('Fibo needs a pinned component store')

    def hub(self, data, variant='base', transfer=None):
        repo, revision = {
            'base': ('briaai/Fibo-2.0-dev', 'cbf184270ff8f2cf9f2daf464564f37fd34c4a05'),
            'turbo': ('briaai/Fibo-2.0-edit-turbo', '63316ff6d5c4c4ba37729a8a6510bced7e8f585b'),
        }[variant]
        owner = self
        hub = ModuleType('huggingface_hub')
        class API:
            def __init__(self, **kwargs):
                pass
            def model_info(self, model, *, revision: str, files_metadata):
                owner.assertEqual(model, repo)
                owner.assertEqual(revision, expected_revision)
                return SimpleNamespace(siblings=[SimpleNamespace(rfilename=k, size=len(v)) for k, v in data.items()])
        expected_revision = revision
        def download(*, repo_id, filename, local_dir, revision, **kwargs):
            owner.assertEqual((repo_id, revision), (repo, expected_revision))
            path = Path(local_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data[filename])
            if transfer:
                transfer(filename, path)
            return str(path)
        hub.HfApi, hub.hf_hub_download = API, download
        return patch.dict(sys.modules, {'huggingface_hub': hub})

    def test_components_include_their_dependencies_and_publish_before_later_weights(self):
        store = self.store()
        with TemporaryDirectory() as directory:
            root, published = Path(directory), []
            owner = self
            class Publisher:
                def ready(self, component, path):
                    published.append(component)
                    if component == 'transformer':
                        owner.assertFalse((root / 'text_encoder/model.safetensors').exists())
                    if component == 'text_encoder':
                        owner.assertTrue((root / 'processor/tokenizer.json').is_file())
                        owner.assertTrue((root / 'fonts/DejaVuSans.ttf').is_file())
                        owner.assertFalse((root / 'vae/diffusion_pytorch_model.safetensors').exists())
                    if component == 'vae':
                        owner.assertTrue((root / 'scheduler/scheduler_config.json').is_file())
            with self.hub(files()):
                store.download_model(root, variant='base', publisher=Publisher())
            self.assertEqual(published, ['transformer', 'text_encoder', 'vae', 'all'])
            self.assertTrue(store.snapshot_ready(root, variant='base'))
            with patch.dict(sys.modules, {'huggingface_hub': None}):
                store.download_model(root, variant='base')

    def test_late_partial_file_fails_then_retry_reuses_completed_files(self):
        store = self.store()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = 'vae/diffusion_pytorch_model.safetensors'
            def truncate(name, path):
                if name == target:
                    path.write_bytes(b'x')
            with self.hub(files(), transfer=truncate), self.assertRaisesRegex(ValueError, 'Incomplete'):
                store.download_model(root, variant='base')
            self.assertFalse(store.snapshot_ready(root, variant='base'))
            def repair_only_target(name, path):
                self.assertEqual(name, target)
            with self.hub(files(), transfer=repair_only_target):
                store.download_model(root, variant='base')
            self.assertTrue(store.snapshot_ready(root, variant='base'))

    def test_variant_provenance_does_not_reuse_same_sized_wrong_weights(self):
        store = self.store()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.hub(files()):
                store.download_model(root, variant='base')
            self.assertFalse(store.snapshot_ready(root, variant='turbo'))
            data = files()
            target = 'transformer/diffusion_pytorch_model-00001-of-00002.safetensors'
            data[target] = b'x' * len(data[target])
            with self.hub(data, variant='turbo'):
                store.download_model(root, variant='turbo')
            self.assertEqual((root / target).read_bytes(), data[target])
            self.assertTrue(store.snapshot_ready(root, variant='turbo'))


if __name__ == '__main__':
    unittest.main()
