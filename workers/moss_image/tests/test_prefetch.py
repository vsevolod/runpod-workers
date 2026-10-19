"""A loader consumes finished components while the downloader fetches the rest."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path[:0] = [str(Path(__file__).parents[2]), str(Path(__file__).parents[1])]


class PrefetchTests(unittest.TestCase):
    def test_adapter_only_download_never_fetches_sft_and_reuses_complete_local_copy(self):
        from moss_image_infer import model_store as store
        from moss_image_infer.config import LORA_ID
        spec = importlib.util.spec_from_file_location('_moss_adapter_download', Path(__file__).parents[1] / 'download_weights.py')
        downloader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(downloader)
        repos = []
        def remote(repo, revision, token):
            repos.append(repo)
            self.assertEqual(repo, LORA_ID)
            return [SimpleNamespace(rfilename='lora_latest.safetensors', size=4)]
        def fetch(session, repo, filename, *, revision):
            path = session.output / filename
            path.write_bytes(b'lora')
            return path
        with tempfile.TemporaryDirectory() as tmp, patch.object(store, 'remote_files', side_effect=remote), patch(
            'shared.model_download.ModelDownloader.file', new=fetch
        ):
            root, state = Path(tmp) / 'model', Path(tmp) / 'state'
            state.mkdir()
            for _ in range(2):
                self.assertEqual(downloader.main(['--output', str(root), '--lora-only', '--prefetch-state-dir', str(state)]), 0)
            self.assertEqual(repos, [LORA_ID])
            self.assertFalse((root / 'transformer').exists())
            self.assertFalse((state / 'transformer.json').exists())
            self.assertTrue((state / 'turbo_lora.json').exists())
            self.assertTrue((state / 'all.json').exists())

    def test_retry_reuses_validated_transformer_after_a_later_download_fails(self):
        from moss_image_infer import model_store as store
        from shared.prefetch import ComponentPublisher
        files = {name: b'complete' for name in ('transformer/config.json', 'text_encoder/config.json', 'vae/config.json')}
        siblings = [SimpleNamespace(rfilename=p, size=len(b)) for p, b in files.items()]
        fetched = []
        interrupted = True
        def fetch(session, repo, filename, *, revision):
            fetched.append(filename)
            if interrupted and filename.startswith('text_encoder/'):
                raise OSError('interrupted encoder')
            path = session.output / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(files[filename])
            return path
        with tempfile.TemporaryDirectory() as tmp, patch.object(store, 'remote_files', return_value=siblings), patch(
            'shared.model_download.ModelDownloader.file', new=fetch
        ):
            root, state = Path(tmp) / 'model', Path(tmp) / 'state'
            state.mkdir()
            with self.assertRaisesRegex(OSError, 'interrupted encoder'):
                store.download_model(root, publisher=ComponentPublisher(state))
            self.assertFalse(store.snapshot_ready(root))
            self.assertFalse((state / 'text_encoder.json').exists())
            interrupted = False
            store.download_model(root, publisher=ComponentPublisher(state))
            self.assertEqual(fetched.count('transformer/config.json'), 1)
            self.assertTrue(store.snapshot_ready(root))

    def test_downloader_publishes_components_before_fetching_the_next_and_turbo_last(self):
        from moss_image_infer import model_store as store
        from moss_image_infer.config import MODEL_ID, LORA_ID
        spec = importlib.util.spec_from_file_location('_moss_download_prefetch', Path(__file__).parents[1] / 'download_weights.py')
        downloader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(downloader)
        files = {name: b'complete' for name in (
            'vae/config.json', 'scheduler/scheduler_config.json', 'tokenizer/tokenizer.json',
            'text_encoder/config.json', 'transformer/config.json', 'transformer/weights.safetensors',
        )}
        def remote(repo, revision, token):
            return [SimpleNamespace(rfilename=p, size=len(b)) for p, b in (
                files.items() if repo == MODEL_ID else [('lora_latest.safetensors', b'complete')])]
        with tempfile.TemporaryDirectory() as tmp:
            root, state = Path(tmp) / 'model', Path(tmp) / 'state'
            state.mkdir()
            def fetch(session, repo, filename, *, revision):
                if filename.startswith(('text_encoder/', 'tokenizer/')):
                    self.assertTrue((state / 'transformer.json').exists())
                if filename.startswith(('vae/', 'scheduler/')):
                    self.assertTrue((state / 'text_encoder.json').exists())
                if repo == LORA_ID:
                    self.assertTrue((state / 'vae.json').exists())
                self.assertFalse((state / 'all.json').exists())
                path = session.output / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'complete')
                return path
            with patch.object(store, 'remote_files', side_effect=remote), patch(
                'shared.model_download.ModelDownloader.file', new=fetch
            ):
                self.assertEqual(downloader.main(['--output', str(root), '--turbo', '--prefetch-state-dir', str(state)]), 0)
            for name in ('transformer', 'text_encoder', 'vae', 'turbo_lora', 'all'):
                self.assertTrue((state / f'{name}.json').exists(), name)

    def test_pipeline_loads_each_component_before_the_next_download_finishes(self):
        from shared.prefetch import PrefetchSession
        from moss_image_infer import pipeline
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / 'download.py'
            script.write_text('''
import argparse, time
from pathlib import Path
from shared.prefetch import ComponentPublisher
p = argparse.ArgumentParser(); p.add_argument('--prefetch-state-dir'); a = p.parse_args()
root = Path(__file__).parent
publisher = ComponentPublisher(a.prefetch_state_dir)
for name in ('transformer', 'text_encoder', 'vae'):
    (root / name).mkdir()
    publisher.ready(name, root / name)
    deadline = time.monotonic() + 5
    while not (root / (name + '-loaded')).exists():
        if time.monotonic() > deadline: raise RuntimeError('Loader waited for later components: ' + name)
        time.sleep(.01)
(root / 'scheduler').mkdir()
publisher.ready('all', root)
''')
            def load(name):
                def component(*args, **kwargs):
                    self.assertTrue((root / name).is_dir(), 'Read before component was ready')
                    following = {'transformer': 'text_encoder', 'text_encoder': 'vae'}.get(name)
                    if following:
                        self.assertFalse((root / following).exists(), 'Loader waited for the entire download')
                    (root / (name + '-loaded')).touch()
                    return Mock()
                return component
            env = {**os.environ, 'PYTHONPATH': str(Path(__file__).parents[2])}
            with PrefetchSession([sys.executable, str(script)], env=env) as pending, patch.dict(
                os.environ, {'MOSS_OFFLOAD': 'none'}
            ), patch.object(pipeline.MOSSImageTransformer, 'from_pretrained', side_effect=load('transformer')), patch.object(
                pipeline.AutoModel, 'from_pretrained', side_effect=load('text_encoder')
            ), patch.object(pipeline.AutoTokenizer, 'from_pretrained', return_value=Mock()), patch.object(
                pipeline.AutoencoderKLFlux2, 'from_pretrained', side_effect=load('vae')
            ), patch.object(pipeline.MOSSImageScheduler, 'from_pretrained', return_value=Mock()), patch.object(
                pipeline, 'MOSSImagePipeline', return_value=Mock()
            ):
                runtime = pipeline.load_pipeline(root, device='cpu')
                pending.wait('all', timeout=5)
                runtime.close()


if __name__ == '__main__':
    unittest.main()
