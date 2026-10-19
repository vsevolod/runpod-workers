"""Keep model-specific downloads, paths and resident owners separate."""

from contextlib import nullcontext
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parents[1]))


class BackendTests(unittest.TestCase):
    def module(self):
        try:
            from image_worker import backends
        except ImportError:
            self.fail('Model backends are missing')
        return backends

    def test_krea_only_downloads_on_load_and_uses_scoped_paths_and_quantization(self):
        module = self.module()
        pending = SimpleNamespace(wait=Mock())
        pipe = SimpleNamespace()
        with patch.dict(os.environ, {
            'MODEL_DIR': '/wrong/model', 'LORA_DIR': '/wrong/loras',
            'KREA_MODEL_DIR': '/models/selected-krea', 'KREA_LORA_DIR': '/volume/krea/loras',
            'KREA_DIT_QUANT': 'int8',
        }, clear=True), patch.object(module, 'PrefetchSession', return_value=nullcontext(pending)) as prefetch, patch.object(
            module, 'load_krea_pipeline', return_value=pipe
        ) as load:
            backend = module.KreaBackend()
            prefetch.assert_not_called()
            load.assert_not_called()
            backend.load()
            command = prefetch.call_args.args[0]
            self.assertEqual(command[command.index('--output') + 1], '/models/selected-krea')
            self.assertEqual(command[command.index('--quant') + 1], 'int8_convrot')
            self.assertIn('--skip-vae-file', command)
            load.assert_called_once_with(model_dir=Path('/models/selected-krea'),
                                         lora_dir=Path('/volume/krea/loras'),
                                         quant_mode='int8_convrot', local_files_only=True)
            self.assertIs(backend.pipe, pipe)
            pending.wait.assert_called_once_with('all')
            backend.close()
            backend.close()
            self.assertIsNone(backend.pipe)

    def test_qwen_owns_and_closes_its_variant_manager(self):
        module = self.module()
        manager = SimpleNamespace(close=Mock())
        with patch.object(module, 'QwenModelManager', return_value=manager) as factory:
            backend = module.QwenBackend()
            factory.assert_not_called()
            backend.load()
            self.assertIs(backend.manager, manager)
            backend.close()
            backend.close()
        manager.close.assert_called_once()
        self.assertIsNone(backend.manager)
