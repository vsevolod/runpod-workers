"""Request-time copies must never expose network mappings or partial adapters."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[2]))


class AdapterCacheTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.find_spec('shared.adapter_cache')
        self.assertIsNotNone(spec, 'Request-time adapter caching is missing')
        from shared.adapter_cache import AdapterCache
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'network'
        self.source.mkdir()
        self.a = self.source / 'a.safetensors'
        self.b = self.source / 'b.safetensors'
        self.a.write_bytes(b'adapter A')
        self.b.write_bytes(b'adapter B')
        self.cache = AdapterCache(self.root / 'local')

    def fetch(self, *paths):
        with self.cache.prefetch(paths) as pending:
            return pending.result()

    def test_only_selected_files_are_copied_and_warm_hits_reuse_them(self):
        (local,) = self.fetch(self.a)
        self.assertEqual(local.read_bytes(), b'adapter A')
        self.assertTrue(local.is_relative_to(self.root / 'local'))
        self.assertNotEqual(local.stat().st_ino, self.a.stat().st_ino)
        self.assertFalse(local.is_symlink())
        modified = local.stat().st_mtime_ns
        self.assertEqual(self.fetch(self.a), (local,))
        self.assertEqual(local.stat().st_mtime_ns, modified)
        self.assertEqual([p.name for p in (self.root / 'local').rglob('*.safetensors')], ['a.safetensors'])

    def test_each_request_restores_missing_or_truncated_local_files(self):
        (local,) = self.fetch(self.a)
        local.unlink()
        self.assertEqual(self.fetch(self.a)[0].read_bytes(), b'adapter A')
        local.write_bytes(b'partial')
        self.assertEqual(self.fetch(self.a)[0].read_bytes(), b'adapter A')
        self.assertEqual(self.fetch(self.b)[0].read_bytes(), b'adapter B')

    def test_replaced_source_of_same_size_is_refreshed(self):
        self.fetch(self.a)
        previous = self.a.stat()
        self.a.write_bytes(b'adapter C')
        # A metadata-based cache needs an observable metadata change, even on
        # filesystems whose timestamps do not advance between these fast writes.
        os.utime(self.a, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))
        self.assertEqual(self.fetch(self.a)[0].read_bytes(), b'adapter C')

    def test_background_copy_overlaps_caller_and_failed_copy_is_retryable(self):
        import shared.adapter_cache as module
        entered, release = threading.Event(), threading.Event()
        def blocked(source, destination):
            Path(destination).write_bytes(b'partial')
            entered.set()
            if not release.wait(3):
                raise TimeoutError('caller did not run concurrently')
            raise OSError('network disconnected')
        with self.assertLogs(module.logger, level='ERROR'), patch.object(module.shutil, 'copyfile', side_effect=blocked):
            with self.cache.prefetch((self.a,)) as pending:
                self.assertTrue(entered.wait(3))
                self.assertFalse(list((self.root / 'local').rglob('*.safetensors')))
                release.set()
                with self.assertRaisesRegex(OSError, 'disconnected'):
                    pending.result()
        self.assertEqual(self.fetch(self.a)[0].read_bytes(), b'adapter A')
        self.assertFalse(list((self.root / 'local').rglob('*.part')))

    def test_source_changed_during_copy_is_not_published(self):
        import shared.adapter_cache as module
        original = module.shutil.copyfile
        def changing(source, destination):
            original(source, destination)
            previous = self.a.stat()
            self.a.write_bytes(b'adapter C')
            os.utime(self.a, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))
        with self.assertLogs(module.logger, level='ERROR'), patch.object(module.shutil, 'copyfile', side_effect=changing):
            with self.assertRaisesRegex(OSError, 'changed'):
                self.fetch(self.a)
        self.assertFalse(list((self.root / 'local').rglob('*.safetensors')))
        self.assertEqual(self.fetch(self.a)[0].read_bytes(), b'adapter C')

    def test_concurrent_requests_get_complete_shared_copy(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            paths = list(executor.map(lambda _: self.fetch(self.a)[0], range(4)))
        self.assertEqual(len(set(paths)), 1)
        self.assertEqual(paths[0].read_bytes(), b'adapter A')
        from shared.adapter_cache import AdapterCache
        self.cache = AdapterCache(self.root / 'local')
        modified = paths[0].stat().st_mtime_ns
        self.assertEqual(self.fetch(self.a)[0].stat().st_mtime_ns, modified)

    def test_empty_selection_has_no_disk_effects(self):
        self.assertEqual(self.fetch(), ())
        self.assertFalse((self.root / 'local').exists())
