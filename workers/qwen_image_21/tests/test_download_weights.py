"""Download real component layouts through a controlled Hub transport."""

import contextlib
import fnmatch
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

_PATH = Path(__file__).parents[1] / "download_weights.py"


def _load():
    spec = importlib.util.spec_from_file_location('_qwen_downloader_under_test', _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Hub(ModuleType):
    def __init__(self):
        super().__init__('huggingface_hub')
        self.calls = []
        self.fail_vae = False
        self.fail_encoder = False
        self.revision = 'a' * 40
        self.versioned_weights = False
        self.invalid_schedule = False
        self.missing_encoder_shard = False
        self.HfApi = lambda **kwargs: SimpleNamespace(model_info=lambda repo: SimpleNamespace(sha=self.revision))

    def snapshot_download(self, **kwargs):
        self.calls.append(kwargs)
        repo, root = kwargs['repo_id'], Path(kwargs['local_dir'])
        root.mkdir(parents=True, exist_ok=True)
        if self.fail_encoder and 'text_encoder/*' in (kwargs.get('allow_patterns') or []):
            raise OSError('Encoder transfer interrupted')
        if repo == 'madebyollin/texture-fix-vae-for-qwen-image-2.1':
            if self.fail_vae:
                raise OSError('VAE transfer interrupted')
            files = {'config.json': b'{}', 'diffusion_pytorch_model.safetensors': b'complete vae'}
        else:
            config = {'_class_name': 'QwenImage21Pipeline'}
            if repo.endswith('-Turbo'):
                config['sample_sigmas'] = [] if self.invalid_schedule else [1, .98, .95, .92, .89, .84, .70, .41]
            files = {'model_index.json': json.dumps(config).encode(),
                     'scheduler/scheduler_config.json': b'{}', 'processor/processor_config.json': b'{}'}
            for name in ('transformer', 'text_encoder', 'vae'):
                files[name + '/config.json'] = b'{}'
                files[name + '/model.safetensors'] = (kwargs['revision'].encode() if self.versioned_weights
                                                     else b'complete component weights')
            if self.missing_encoder_shard:
                files['text_encoder/model.safetensors.index.json'] = json.dumps({
                    'weight_map': {'one': 'missing-shard.safetensors'}}).encode()
        patterns = kwargs.get('allow_patterns')
        for name, content in files.items():
            if patterns is None or any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        return str(root)


class DownloadWeightsTests(unittest.TestCase):
    def test_retry_does_not_mix_completed_weights_with_a_new_upstream_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, hub = Path(tmp), Hub()
            hub.versioned_weights = True
            hub.fail_encoder = True
            self.assertEqual(self.run_download(root, hub, state=root / 'first'), 1)
            self.assertEqual((root / 'model/transformer/model.safetensors').read_bytes(), b'a' * 40)
            hub.fail_encoder = False
            hub.revision = 'b' * 40
            self.assertEqual(self.run_download(root, hub, state=root / 'retry'), 0)
            self.assertEqual((root / 'model/transformer/model.safetensors').read_bytes(), b'a' * 40)
            self.assertEqual((root / 'model/text_encoder/model.safetensors').read_bytes(), b'a' * 40)

    @classmethod
    def setUpClass(cls):
        cls.mod = _load()

    def run_download(self, root, hub, *, variant='turbo', vae='original', state=None):
        args = ['--model-variant', variant, '--output', str(root / 'model'), '--token', 'hf_test']
        if state is not None:
            state.mkdir(exist_ok=True)
            args.extend(['--prefetch-state-dir', str(state)])
        with patch.dict(sys.modules, {'huggingface_hub': hub}), patch.dict(os.environ, {'QWEN_VAE': vae}), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return self.mod.main(args)

    def test_downloads_only_selected_variant_with_its_saved_schedule(self):
        for variant, repo in (('base', 'Qwen/Qwen-Image-2.1'), ('turbo', 'Qwen/Qwen-Image-2.1-Turbo')):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                root, hub = Path(tmp), Hub()
                self.assertEqual(self.run_download(root, hub, variant=variant), 0)
                self.assertEqual({call['repo_id'] for call in hub.calls}, {repo})
                self.assertTrue((root / 'model/transformer/model.safetensors').exists())
                self.assertTrue((root / 'model/text_encoder/model.safetensors').exists())
                config = json.loads((root / 'model/model_index.json').read_text())
                self.assertEqual('sample_sigmas' in config, variant == 'turbo')
                self.assertTrue(all(call['token'] == 'hf_test' for call in hub.calls))

    def test_component_transfers_share_one_pinned_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, hub = Path(tmp), Hub()
            self.assertEqual(self.run_download(root, hub, state=root / 'state'), 0)
            self.assertEqual([call['revision'] for call in hub.calls], ['a' * 40] * 4)
            self.assertTrue((root / 'state/all.json').exists())

    def test_bad_turbo_metadata_never_releases_weights_or_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, hub = Path(tmp), Hub()
            hub.invalid_schedule = True
            self.assertEqual(self.run_download(root, hub, state=root / 'state'), 1)
            self.assertFalse((root / 'state/config.json').exists())
            self.assertFalse((root / 'state/all.json').exists())
            self.assertEqual(len(hub.calls), 1)

    def test_old_snapshot_is_rejected_before_contacting_hub(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, hub = Path(tmp), Hub()
            (root / 'model').mkdir()
            (root / 'model/model_index.json').write_text('{}')
            self.assertEqual(self.run_download(root, hub), 1)
            self.assertEqual(hub.calls, [])

    def test_texture_fix_skips_unused_original_vae(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, hub = Path(tmp), Hub()
            self.assertEqual(self.run_download(root, hub, vae='texture-fix', state=root / 'state'), 0)
            self.assertEqual([call['allow_patterns'] for call in hub.calls[:3]], [
                ['model_index.json', 'scheduler/*', 'processor/*'], ['transformer/*'], ['text_encoder/*']])
            self.assertEqual(hub.calls[-1]['repo_id'], 'madebyollin/texture-fix-vae-for-qwen-image-2.1')
            self.assertFalse((root / 'model/vae').exists())
            self.assertTrue((root / 'model/texture_fix_vae/diffusion_pytorch_model.safetensors').exists())
            self.assertTrue((root / 'state/vae.json').exists())

    def test_warm_complete_snapshot_performs_no_hub_transfers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, hub = Path(tmp), Hub()
            self.assertEqual(self.run_download(root, hub, vae='texture-fix'), 0)
            hub.calls.clear()
            self.assertEqual(self.run_download(root, hub, vae='texture-fix', state=root / 'state'), 0)
            self.assertEqual(hub.calls, [])
            self.assertTrue((root / 'state/all.json').exists())

    def test_truncated_transformer_is_repaired_without_refetching_encoder_or_vae(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, hub = Path(tmp), Hub()
            self.assertEqual(self.run_download(root, hub, vae='texture-fix'), 0)
            (root / 'model/transformer/model.safetensors').write_bytes(b'partial')
            hub.calls.clear()
            self.assertEqual(self.run_download(root, hub, vae='texture-fix'), 0)
            self.assertEqual([call['allow_patterns'] for call in hub.calls], [
                ['model_index.json', 'scheduler/*', 'processor/*'], ['transformer/*']])
            self.assertEqual((root / 'model/transformer/model.safetensors').read_bytes(), b'complete component weights')

    def test_truncated_known_metadata_is_repaired_without_refetching_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, hub = Path(tmp), Hub()
            self.assertEqual(self.run_download(root, hub), 0)
            (root / 'model/model_index.json').write_bytes(b'{')
            hub.calls.clear()
            self.assertEqual(self.run_download(root, hub), 0)
            self.assertEqual([call['allow_patterns'] for call in hub.calls], [
                ['model_index.json', 'scheduler/*', 'processor/*']])
            self.assertEqual(json.loads((root / 'model/model_index.json').read_text())['_class_name'], 'QwenImage21Pipeline')

    def test_missing_encoder_shard_does_not_release_encoder_or_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, hub = Path(tmp), Hub()
            hub.missing_encoder_shard = True
            self.assertEqual(self.run_download(root, hub, state=root / 'state'), 1)
            self.assertTrue((root / 'state/transformer.json').exists())
            self.assertFalse((root / 'state/text_encoder.json').exists())
            self.assertFalse((root / 'state/all.json').exists())

    def test_vae_failure_prevents_final_ready_and_retry_reuses_completed_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, hub = Path(tmp), Hub()
            hub.fail_vae = True
            self.assertEqual(self.run_download(root, hub, vae='texture-fix', state=root / 'state'), 1)
            self.assertTrue((root / 'state/text_encoder.json').exists())
            self.assertFalse((root / 'state/vae.json').exists())
            self.assertFalse((root / 'state/all.json').exists())
            hub.fail_vae = False
            hub.calls.clear()
            self.assertEqual(self.run_download(root, hub, vae='texture-fix', state=root / 'retry'), 0)
            self.assertEqual([call['repo_id'] for call in hub.calls], ['madebyollin/texture-fix-vae-for-qwen-image-2.1'])

    def test_help_exits_cleanly(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
            self.mod.main(['--help'])
        self.assertEqual(result.exception.code, 0)


if __name__ == '__main__':
    unittest.main()
