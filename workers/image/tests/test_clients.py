"""Both model clients target the common endpoint and explicitly select a model."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).parents[3]


class ClientTests(unittest.TestCase):
    def client(self, name):
        filename = 'krea2_image.py' if name == 'krea2' else 'qwen_image_21.py'
        spec = importlib.util.spec_from_file_location('_unified_' + name + '_client', ROOT / 'scripts' / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_generate_and_edit_payloads_select_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'reference.png'
            Image.new('RGB', (64, 96)).save(image)
            for name in ('krea2', 'qwen_image_21'):
                client = self.client(name)
                for edit in (False, True):
                    with self.subTest(model=name, edit=edit):
                        command = ['--prompt', 'fox', '--lora', '', '--seed', '0']
                        if edit:
                            command.extend(['--image', str(image)])
                        args = client._build_parser().parse_args(command)
                        with patch.object(client, '_submit_job', return_value='job') as submit, patch.object(
                            client, '_poll_job', return_value={}
                        ), patch.object(client, '_save_result', return_value=0):
                            run = client._run_edit if edit else client._run_generate
                            self.assertEqual(run(args, endpoint_id='common', api_key='key'), 0)
                        self.assertEqual(submit.call_args.args[2]['model'], name)
                        self.assertEqual(submit.call_args.args[2]['seed'], 0)

    def test_both_default_to_the_retained_endpoint_and_allow_an_override(self):
        for name in ('krea2', 'qwen_image_21'):
            client = self.client(name)
            for override, expected in (('', '9zb0wyo61ck3wk'), ('other-endpoint', 'other-endpoint')):
                with self.subTest(model=name, override=override), patch.dict(
                    os.environ, {'RUNPOD_API_KEY': 'key', 'ENDPOINT_ID': override}, clear=True
                ), patch.object(sys, 'argv', ['client', '--prompt', 'fox']), patch.object(
                    client, '_run_generate', return_value=0
                ) as generate:
                    self.assertEqual(client.main(), 0)
                    self.assertEqual(generate.call_args.kwargs['endpoint_id'], expected)
