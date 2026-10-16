"""Exercise readiness and cleanup with real downloader subprocesses."""

import importlib
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


WORKERS = Path(__file__).parents[2]
sys.path.insert(0, str(WORKERS))


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((WORKERS / "shared/prefetch.py").is_file(), "Prefetch coordination is missing")
        self.mod = importlib.import_module("shared.prefetch")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def session(self, body):
        script = self.root / "download.py"
        script.write_text(textwrap.dedent('''
            import argparse, time
            from pathlib import Path
            from shared.prefetch import ComponentPublisher
            parser = argparse.ArgumentParser()
            parser.add_argument('--prefetch-state-dir')
            args = parser.parse_args()
            publisher = ComponentPublisher(args.prefetch_state_dir)
            root = Path(__file__).parent
        ''') + textwrap.dedent(body))
        env = {**os.environ, "PYTHONPATH": str(WORKERS)}
        return self.mod.PrefetchSession([sys.executable, "-u", str(script)], env=env)

    def test_loader_can_use_dit_while_vae_download_is_still_pending(self):
        session = self.session('''
            dit = root / 'dit.safetensors'
            dit.write_bytes(b'complete dit')
            publisher.ready('dit', dit)
            deadline = time.monotonic() + 5
            while not (root / 'dit_loaded').exists():
                if time.monotonic() > deadline: raise RuntimeError('Loader never consumed DiT')
                time.sleep(.01)
            vae = root / 'vae'
            vae.mkdir()
            publisher.ready('vae', vae)
            publisher.ready('all', root)
        ''')
        with session:
            dit = self.mod.wait_for_component("dit", timeout=3)
            self.assertEqual(dit.read_bytes(), b"complete dit")
            self.assertIsNone(session.process.poll())
            self.assertFalse((self.root / "vae").exists())
            (self.root / "dit_loaded").touch()
            self.assertTrue(session.wait("vae", timeout=3).is_dir())
            session.wait("all", timeout=3)
        self.assertEqual(session.process.returncode, 0)

    def test_incomplete_file_is_not_exposed_before_ready_signal(self):
        session = self.session('''
            partial = root / 'dit.incomplete'
            partial.write_bytes(b'partial')
            deadline = time.monotonic() + 5
            while not (root / 'finish').exists():
                if time.monotonic() > deadline: raise RuntimeError('No finish signal')
                time.sleep(.01)
            dit = root / 'dit.safetensors'
            dit.write_bytes(b'complete')
            publisher.ready('dit', dit)
            publisher.ready('all', root)
        ''')
        with session:
            with self.assertRaises(TimeoutError):
                session.wait("dit", timeout=.15)
            (self.root / "finish").touch()
            self.assertEqual(session.wait("dit", timeout=3).read_bytes(), b"complete")
            session.wait("all", timeout=3)

    def test_downloader_failure_after_dit_prevents_serving_partial_pipeline(self):
        session = self.session('''
            dit = root / 'dit.safetensors'
            dit.touch()
            publisher.ready('dit', dit)
            raise SystemExit(7)
        ''')
        with session:
            self.assertTrue(session.wait("dit", timeout=3).is_file())
            with self.assertRaisesRegex(self.mod.DownloadError, "7"):
                session.wait("vae", timeout=3)
            with self.assertRaises(self.mod.DownloadError):
                session.wait("all", timeout=3)

    def test_missing_component_fails_even_when_downloader_exits_successfully(self):
        with self.session("pass") as session:
            with self.assertRaisesRegex(self.mod.DownloadError, "dit"):
                session.wait("dit", timeout=3)

    def test_all_ready_does_not_hide_a_nonzero_exit(self):
        with self.session("publisher.ready('all', root)\nraise SystemExit(9)") as session:
            with self.assertRaisesRegex(self.mod.DownloadError, "9"):
                session.wait("all", timeout=3)

    def test_loading_error_terminates_downloader_and_removes_stale_readiness(self):
        session = self.session('''
            dit = root / 'dit.safetensors'
            dit.touch()
            publisher.ready('dit', dit)
            while True: time.sleep(.1)
        ''')
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            with session:
                session.wait("dit", timeout=3)
                state = session.state_dir
                raise RuntimeError("load failed")
        self.assertIsNotNone(session.process.poll())
        self.assertFalse(state.exists())
        self.assertIsNone(self.mod.wait_for_component("dit"))


if __name__ == "__main__":
    unittest.main()
