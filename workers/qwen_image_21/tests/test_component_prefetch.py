"""Qwen must release and load components before the whole snapshot is present."""

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


class ComponentPrefetchTests(unittest.TestCase):
    def test_downloader_releases_transformer_before_downloading_text_encoder(self):
        spec = importlib.util.spec_from_file_location('_qwen_component_download', Path(__file__).parents[1] / 'download_weights.py')
        downloader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(downloader)
        with tempfile.TemporaryDirectory() as tmp:
            root, state = Path(tmp) / 'model', Path(tmp) / 'state'
            state.mkdir()
            def snapshot(session, repo, *, local_dir, allow_patterns):
                path = Path(local_dir)
                path.mkdir(parents=True, exist_ok=True)
                if 'text_encoder/*' in allow_patterns:
                    self.assertTrue((state / 'transformer.json').exists(), 'Encoder transfer started before transformer release')
                if repo == 'Qwen/Qwen-Image-2.1':
                    (path / 'model_index.json').write_text(json.dumps({'_class_name': 'QwenImage21Pipeline'}))
                    for name in ('transformer', 'text_encoder'):
                        if name + '/*' in allow_patterns:
                            folder = path / name
                            folder.mkdir(exist_ok=True)
                            (folder / 'config.json').write_text('{}')
                            (folder / 'model.safetensors').write_bytes(b'complete weights')
                else:
                    self.assertTrue((state / 'text_encoder.json').exists())
                    (path / 'config.json').write_text('{}')
                    (path / 'diffusion_pytorch_model.safetensors').write_bytes(b'complete vae')
                return path
            with patch('shared.model_download.ModelDownloader.snapshot', new=snapshot), patch.dict(
                os.environ, {'QWEN_VAE': 'texture-fix'}
            ):
                self.assertEqual(downloader.main(['--model-variant', 'base', '--output', str(root),
                                                 '--prefetch-state-dir', str(state)]), 0)
            for name in ('config', 'transformer', 'text_encoder', 'vae', 'all'):
                self.assertTrue((state / f'{name}.json').exists(), name)

    def test_staged_loader_waits_for_each_component_and_passes_loaded_weights_to_pipeline(self):
        from qwen_image_21_infer import pipeline
        from shared.prefetch import PrefetchSession
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
publisher.ready('all', root)
''')
            components = {}
            def component(name):
                def load(*args, **kwargs):
                    self.assertTrue((root / name).is_dir(), 'Read before component was ready')
                    following = {'transformer': 'text_encoder', 'text_encoder': 'vae'}.get(name)
                    if following:
                        self.assertFalse((root / following).exists())
                    (root / (name + '-loaded')).touch()
                    components[name] = object()
                    return components[name]
                return SimpleNamespace(from_pretrained=load)
            def assemble(source, **kwargs):
                self.assertEqual(source, str(root))
                for name in ('transformer', 'text_encoder', 'vae'):
                    self.assertIs(kwargs[name], components[name], 'Pipeline must not reload weights')
                return 'assembled'
            env = {**os.environ, 'PYTHONPATH': str(Path(__file__).parents[2])}
            import diffusers, transformers
            with PrefetchSession([sys.executable, str(script)], env=env) as pending, patch.object(
                diffusers.QwenImage21Transformer2DModel, 'from_pretrained', side_effect=component('transformer').from_pretrained
            ), patch.object(transformers.Qwen3VLForConditionalGeneration, 'from_pretrained',
                            side_effect=component('text_encoder').from_pretrained), patch.object(
                diffusers.AutoencoderKLQwenImage21, 'from_pretrained', side_effect=component('vae').from_pretrained
            ):
                result = pipeline._load_base_pipeline(SimpleNamespace(from_pretrained=assemble), str(root),
                    local_files_only=True, disable_mmap=False, vae_source=str(root / 'vae'))
                pending.wait('all', timeout=5)
            self.assertEqual(result, 'assembled')


if __name__ == '__main__':
    unittest.main()
