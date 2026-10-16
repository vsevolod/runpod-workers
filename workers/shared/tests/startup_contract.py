"""Run each real entrypoint against tiny download/loader stand-ins."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SHARED = Path(__file__).parents[1]


class StartupContractTests(unittest.TestCase):
    def run_start(self, root, *, download_exit=0, load_error=False):
        app = root / "app"
        app.mkdir()
        shutil.copy(self.start, app / "start.sh")
        shutil.copytree(SHARED, app / "shared", ignore=shutil.ignore_patterns("tests", "__pycache__"))
        (app / "download_weights.py").write_text('''
import argparse, json, os, time
from pathlib import Path
from shared.prefetch import ComponentPublisher
p = argparse.ArgumentParser()
p.add_argument('--prefetch-state-dir')
p.add_argument('--skip-vae-file', action='store_true')
args = p.parse_args()
root = Path(os.environ['TEST_ROOT'])
(root / 'downloader_pid').write_text(str(os.getpid()))
(root / 'download_env.json').write_text(json.dumps({k: os.environ.get(k) for k in ("MODEL_DIR", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "LORA_DIR", "DIT_QUANT", "LOCAL_FILES_ONLY")}))
deadline = time.monotonic() + 3
while not (root / 'imports_started').exists():
    if time.monotonic() > deadline: raise RuntimeError('Imports did not overlap download')
    time.sleep(.01)
if int(os.environ['DOWNLOAD_EXIT']): raise SystemExit(int(os.environ['DOWNLOAD_EXIT']))
publisher = ComponentPublisher(args.prefetch_state_dir)
dit = root / 'dit.safetensors'
dit.write_bytes(b'complete')
publisher.ready('dit', dit)
while os.environ['LOAD_ERROR'] == '1' or (os.environ['PROFILE'] == 'krea2' and not (root / 'dit_loaded').exists()):
    if time.monotonic() > deadline: raise RuntimeError('DiT was never loaded')
    time.sleep(.01)
publisher.ready('vae', root)
publisher.ready('text_encoder', root)
publisher.ready('all', root)
''')
        (app / "handler.py").write_text('''
import json, os
from pathlib import Path
from shared.prefetch import wait_for_component
root = Path(os.environ['TEST_ROOT'])
(root / 'imports_started').touch()
if os.environ['LOAD_ERROR'] == '1':
    wait_for_component('dit')
    raise RuntimeError('load failed')
if os.environ['PROFILE'] == 'krea2':
    dit = wait_for_component('dit')
    assert dit.read_bytes() == b'complete'
    (root / 'dit_loaded').touch()
    wait_for_component('vae')
    wait_for_component('text_encoder')
wait_for_component('all')
(root / 'handler_env.json').write_text(json.dumps({k: os.environ.get(k) for k in ("MODEL_DIR", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "LORA_DIR", "DIT_QUANT", "LOCAL_FILES_ONLY")}))
(root / 'served').touch()
''')
        env = {
            **os.environ, "TEST_ROOT": str(root), "PROFILE": self.profile,
            "MODEL_DIR": str(root / "models"), "HF_HOME": str(root / "hf"),
            "HF_HUB_CACHE": "/old/remote", "HUGGINGFACE_HUB_CACHE": "/old/remote",
            "LORA_DIR": "/runpod-volume/adapters", "DIT_QUANT": "int8_convrot",
            "LOCAL_FILES_ONLY": "0", "DOWNLOAD_EXIT": str(download_exit),
            "LOAD_ERROR": "1" if load_error else "0",
        }
        return subprocess.run(["sh", str(app / "start.sh")], env=env, capture_output=True, text=True, timeout=10)

    def test_imports_overlap_download_and_serving_uses_only_completed_local_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_start(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "served").is_file())
            for name in ["download_env.json", "handler_env.json"]:
                env = json.loads((root / name).read_text())
                self.assertEqual(env["MODEL_DIR"], str(root / "models"))
                self.assertEqual(env["HF_HUB_CACHE"], str(root / "hf/hub"))
                self.assertEqual(env["HUGGINGFACE_HUB_CACHE"], str(root / "hf/hub"))
                self.assertEqual(env["LORA_DIR"], "/runpod-volume/adapters")
                self.assertEqual(env["DIT_QUANT"], "int8_convrot")
            self.assertEqual(env["LOCAL_FILES_ONLY"], "1")

    def test_download_failure_never_starts_serving_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_start(root, download_exit=7)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "served").exists())

    def test_model_loading_failure_reaps_the_background_downloader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_start(root, load_error=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "served").exists())
            pid = int((root / "downloader_pid").read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
