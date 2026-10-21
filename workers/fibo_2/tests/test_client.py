"""The CLI sends JSON objects and ordered references without changing defaults."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image


class ClientTests(unittest.TestCase):
    def test_json_file_edit_payload_and_server_defaults(self):
        path = Path(__file__).parents[3] / 'scripts/fibo_2.py'
        self.assertTrue(path.is_file(), 'Fibo JSON client is missing')
        spec = importlib.util.spec_from_file_location('_fibo_client', path)
        client = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(client)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompt = {'edit_instruction': 'Use the second reference color.'}
            (root / 'prompt.json').write_text(json.dumps(prompt))
            for name, color in [('first', 'red'), ('second', 'blue')]:
                Image.new('RGB', (16, 16), color).save(root / f'{name}.png')
            args = client._build_parser().parse_args(['--prompt-file', str(root / 'prompt.json'),
                '--image', str(root / 'first.png'), '--image', str(root / 'second.png'),
                '--variant', 'turbo', '--seed', '0'])
            with patch.object(client, '_submit_job', return_value='job') as submit, patch.object(
                client, '_poll_job', return_value={}
            ), patch.object(client, '_save_result', return_value=0):
                self.assertEqual(client._run_generate(args, endpoint_id='endpoint', api_key='key'), 0)
            values = submit.call_args.args[2]
            self.assertEqual(values['prompt'], prompt)
            self.assertEqual((values['model'], values['type'], values['seed']), ('fibo_2', 'image_edit', 0))
            self.assertEqual(values['model_options'], {'variant': 'turbo'})
            self.assertEqual(values['images'], [client._image_to_data_url(root / f'{name}.png')
                                               for name in ('first', 'second')])
            for name in ('num_inference_steps', 'guidance_scale', 'width', 'height'):
                self.assertNotIn(name, values)
