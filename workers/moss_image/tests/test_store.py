"""Pinned model files and failed downloads must never look complete."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).parents[2]), str(Path(__file__).parents[1])]


class StoreTests(unittest.TestCase):
    def test_complete_download_is_reused_and_truncated_files_are_repaired(self):
        from moss_image_infer import model_store as store
        from moss_image_infer.config import MODEL_ID, MODEL_REVISION
        files = {'transformer/config.json': b'{}', 'transformer/weights.safetensors': b'weights'}
        siblings = [SimpleNamespace(rfilename=p, size=len(b)) for p, b in files.items()]
        siblings += [SimpleNamespace(rfilename='pipeline_moss_image.py', size=99)]
        fetched = []
        def download(session, repo, filename, *, revision):
            self.assertEqual((repo, revision), (MODEL_ID, MODEL_REVISION))
            fetched.append(filename)
            path = session.output / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(files[filename])
            return path
        with tempfile.TemporaryDirectory() as tmp, patch.object(store, 'remote_files', return_value=siblings), patch(
            'shared.model_download.ModelDownloader.file', new=download
        ):
            root = Path(tmp)
            store.download_model(root)
            self.assertTrue(store.snapshot_ready(root))
            self.assertEqual(sorted(fetched), sorted(files))
            fetched.clear()
            store.download_model(root)
            self.assertEqual(fetched, [])
            (root / 'transformer/weights.safetensors').write_bytes(b'x')
            self.assertFalse(store.snapshot_ready(root))
            store.download_model(root)
            self.assertEqual(fetched, ['transformer/weights.safetensors'])
            self.assertTrue(store.snapshot_ready(root))

    def test_failed_or_wrong_size_transfer_is_not_published(self):
        from moss_image_infer import model_store as store
        siblings = [SimpleNamespace(rfilename='transformer/config.json', size=10)]
        def incomplete(session, *args, **kwargs):
            path = session.output / 'transformer/config.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'{}')
            return path
        with tempfile.TemporaryDirectory() as tmp, patch.object(store, 'remote_files', return_value=siblings), patch(
            'shared.model_download.ModelDownloader.file', new=incomplete
        ):
            with self.assertRaises(ValueError):
                store.download_model(Path(tmp))
            self.assertFalse(store.snapshot_ready(Path(tmp)))

    def test_turbo_downloads_only_the_pinned_adapter_and_repairs_it(self):
        from moss_image_infer import model_store as store
        from moss_image_infer.config import LORA_ID, LORA_REVISION
        files = [SimpleNamespace(rfilename='lora_latest.safetensors', size=4),
                 SimpleNamespace(rfilename='generator.safetensors', size=100)]
        fetched = []
        def download(session, repo, filename, *, revision):
            fetched.append((repo, filename, revision))
            path = session.output / filename
            path.write_bytes(b'lora')
            return path
        with tempfile.TemporaryDirectory() as tmp, patch.object(store, 'remote_files', return_value=files), patch(
            'shared.model_download.ModelDownloader.file', new=download
        ):
            path = store.ensure_turbo_lora(Path(tmp))
            self.assertEqual(fetched, [(LORA_ID, 'lora_latest.safetensors', LORA_REVISION)])
            store.ensure_turbo_lora(Path(tmp))
            self.assertEqual(len(fetched), 1)
            path.write_bytes(b'x')
            store.ensure_turbo_lora(Path(tmp))
            self.assertEqual(len(fetched), 2)

