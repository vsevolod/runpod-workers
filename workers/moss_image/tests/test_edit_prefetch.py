"""Exercise the real downloader in CPU subprocesses with explicit loader gates."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest import TestCase
from unittest.mock import Mock, patch

sys.path[:0] = [str(Path(__file__).parents[2]), str(Path(__file__).parents[1])]


CHILD = '''
import importlib.util, json, sys, time
from pathlib import Path
from types import SimpleNamespace
from shared.prefetch import ComponentPublisher
from shared.model_download import ModelDownloader
from moss_image_infer import model_store as store
from moss_image_infer.config import MODEL_ID, EDIT_ID
mode = sys.argv.pop(1)
root = Path(__file__).parent
files = ('transformer/config.json', 'text_encoder/config.json', 'tokenizer/tokenizer.json',
         'vae/config.json', 'scheduler/scheduler_config.json')
def remote(repo, revision, token):
    assert 'torch' not in sys.modules, 'Downloader imported Torch'
    assert repo in (MODEL_ID, EDIT_ID), 'Downloaded an unselected adapter'
    return [SimpleNamespace(rfilename=p, size=4) for p in
            (files if repo == MODEL_ID else ('adapter_config.json', 'adapter_model.safetensors'))]
def fetch(session, repo, filename, *, revision):
    with (root / 'fetches.jsonl').open('a') as log:
        log.write(json.dumps([repo, revision, filename]) + '\\n')
    path = session.output / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'fail-adapter-once' and filename == 'adapter_model.safetensors' and not (root / 'failed-once').exists():
        path.write_bytes(b'x')
        (root / 'failed-once').touch()
        raise RuntimeError('Interrupted Edit weights transfer')
    path.write_bytes(b'data')
    return path
ready = ComponentPublisher.ready
def gated(self, component, path):
    ready(self, component, path)
    if component == 'all': return
    deadline = time.monotonic() + 10
    while not (root / (component + '-loaded')).exists():
        if time.monotonic() > deadline: raise RuntimeError('Loader waited for a later component: ' + component)
        time.sleep(.01)
store.remote_files = remote
ModelDownloader.file = fetch
ComponentPublisher.ready = gated
spec = importlib.util.spec_from_file_location('downloader', str(Path(store.__file__).parents[1] / 'download_weights.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.main()
if mode == 'bad-exit': raise SystemExit(7)
'''


class EditPrefetchTests(TestCase):
    def load(self, root, mode='ok', *, fail_loading=False, lora_only=False):
        from shared.prefetch import PrefetchSession
        from moss_image_infer import pipeline
        script = root / 'download.py'
        script.write_text(CHILD)
        command = [sys.executable, str(script), mode, '--output', str(root / 'model'), '--edit']
        if lora_only:
            command.append('--lora-only')
        env = {**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path[:2])}
        pending = PrefetchSession(command, env=env)
        self.pending = pending
        def component(name):
            def load(path, **kwargs):
                self.assertTrue(kwargs['local_files_only'])
                self.assertEqual((Path(path) / 'config.json').read_bytes(), b'data')
                if fail_loading:
                    raise ValueError('Transformer loading failed')
                next_component = {'transformer': 'text_encoder', 'text_encoder': 'vae', 'vae': 'edit_lora'}[name]
                self.assertFalse((pending.state_dir / (next_component + '.json')).exists())
                (root / (name + '-loaded')).touch()
                return Mock()
            return load
        def adapter(directory, model):
            self.assertEqual((Path(directory) / 'adapter_model.safetensors').read_bytes(), b'data')
            self.assertEqual((Path(directory) / 'adapter_config.json').read_bytes(), b'data')
            self.assertFalse((pending.state_dir / 'all.json').exists())
            (root / 'edit_lora-loaded').touch()
            return {}
        pipe = Mock()
        pipe.remove_all_hooks.side_effect = lambda: (root / 'closed').touch()
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {'MOSS_OFFLOAD': 'none'}))
            stack.enter_context(patch.object(pipeline.MOSSImageTransformer, 'from_pretrained', side_effect=component('transformer')))
            stack.enter_context(patch.object(pipeline.AutoModel, 'from_pretrained', side_effect=component('text_encoder')))
            stack.enter_context(patch.object(pipeline.AutoTokenizer, 'from_pretrained', return_value=Mock()))
            stack.enter_context(patch.object(pipeline.AutoencoderKLFlux2, 'from_pretrained', side_effect=component('vae')))
            stack.enter_context(patch.object(pipeline.MOSSImageScheduler, 'from_pretrained', return_value=Mock()))
            stack.enter_context(patch.object(pipeline, 'MOSSImagePipeline', return_value=pipe))
            stack.enter_context(patch.object(pipeline, 'load_edit_adapter', side_effect=adapter))
            with pending:
                if lora_only:
                    runtime = pipeline.Runtime(pipe, root / 'model')
                    runtime.load_edit()
                    pending.wait('all', timeout=10)
                else:
                    runtime = pipeline.load_pipeline(root / 'model', device='cpu', edit=True)
                runtime.close()

    def test_edit_loads_early_components_while_later_downloads_are_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.load(root)
            self.assertEqual(self.pending.process.returncode, 0)
            fetched = [json.loads(line) for line in (root / 'fetches.jsonl').read_text().splitlines()]
            self.assertEqual(fetched[-1], ['chinoll/Moss-Image-Edit',
                                         '3be9c40037e1b89b17aa85ff6f8ee805872aa558', 'adapter_model.safetensors'])
            self.assertTrue((root / 'closed').exists())

    def test_later_adapter_failure_preserves_sft_and_retry_repairs_only_edit(self):
        from shared.prefetch import DownloadError
        from moss_image_infer import model_store as store
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(DownloadError):
                self.load(root, 'fail-adapter-once')
            self.assertIsNotNone(self.pending.process.returncode)
            self.assertTrue((root / 'closed').exists())
            self.assertTrue(store.snapshot_ready(root / 'model'))
            self.assertFalse(store.edit_ready(root / 'model'))
            self.assertEqual((root / 'model/edit/adapter_model.safetensors').read_bytes(), b'x')
            self.load(root, lora_only=True)
            self.assertTrue(store.edit_ready(root / 'model'))
            fetched = [json.loads(line) for line in (root / 'fetches.jsonl').read_text().splitlines()]
            self.assertEqual(sum(line[2] == 'transformer/config.json' for line in fetched), 1)
            self.assertEqual(sum(line[2] == 'adapter_model.safetensors' for line in fetched), 2)
            # Warm validated bundle: no child process and no remote metadata call.
            with patch.object(store, 'remote_files', side_effect=AssertionError('Warm request contacted Hub')):
                with store.prefetch_model(root / 'model', edit=True, lora_only=True) as pending:
                    self.assertIsNone(pending)

    def test_bad_exit_after_all_readiness_still_fails_and_releases_runtime(self):
        from shared.prefetch import DownloadError
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(DownloadError):
                self.load(root, 'bad-exit')
            self.assertEqual(self.pending.process.returncode, 7)
            self.assertTrue((root / 'closed').exists())

    def test_loading_failure_terminates_and_reaps_blocked_downloader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'Transformer loading failed'):
                self.load(root, fail_loading=True)
            self.assertIsNotNone(self.pending.process.returncode)
            self.assertFalse((root / 'model/edit').exists())
