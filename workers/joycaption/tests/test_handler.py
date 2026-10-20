"""Caption API contract without downloading a model or initializing CUDA."""

import base64
from contextlib import nullcontext
import importlib.util
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image


@pytest.fixture
def runtime():
    calls = {}

    class Inputs(dict):
        def to(self, device):
            calls['input_device'] = device
            return self

    class Processor:
        image_processor = SimpleNamespace(size={})

        def apply_chat_template(self, messages, **kwargs):
            calls['messages'] = messages
            calls['template'] = kwargs
            return 'rendered image prompt'

        def __call__(self, **kwargs):
            calls['processor'] = kwargs
            return Inputs(input_ids=SimpleNamespace(shape=(1, 3)))

        def batch_decode(self, tokens, **kwargs):
            calls['decoded'] = tokens
            return [' A red square. ']

    processor = Processor()

    class Model:
        device = 'cuda:0'

        def generate(self, **kwargs):
            calls['generate'] = kwargs
            return [[11, 12, 13, 41, 42]]

    def validate(data, schema):
        # GPU/model calls are faked; request decoding and response are real.
        result = {key: value.get('default') for key, value in schema.items()}
        result.update(data)
        return {'validated_input': result}

    fake_torch = SimpleNamespace(
        inference_mode=nullcontext,
        cuda=SimpleNamespace(OutOfMemoryError=type('FakeOOM', (Exception,), {})),
    )
    modules = {
        'torch': fake_torch,
        'runpod': SimpleNamespace(serverless=SimpleNamespace(start=lambda _: None)),
        'runpod.serverless.utils.rp_validator': SimpleNamespace(validate=validate),
    }
    spec = importlib.util.spec_from_file_location('caption_handler_test', Path(__file__).parents[1] / 'handler.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules), patch.dict('os.environ', {}, clear=True):
        spec.loader.exec_module(module)
        module.MODELS = SimpleNamespace(processor=processor, model=Model(),
                                        model_id='prithivMLmods/Gliese-Qwen3.5-9B-Abliterated-Caption')
        yield module, calls, processor


def image_job(**kwargs):
    buffer = BytesIO()
    Image.new('RGB', (8, 8), 'red').save(buffer, format='PNG')
    return {'input': {'image': base64.b64encode(buffer.getvalue()).decode(), **kwargs}}


def test_caption_receives_image_disables_thinking_and_trims_input_tokens(runtime):
    module, calls, _ = runtime
    result = module.handler(image_job(prompt='Describe the colors.'))
    assert result['caption'] == 'A red square.'
    assert result['prompt'] == 'Describe the colors.'
    assert result['model'] == 'prithivMLmods/Gliese-Qwen3.5-9B-Abliterated-Caption'
    assert calls['messages'] == [{'role': 'user', 'content': [
        {'type': 'image'}, {'type': 'text', 'text': 'Describe the colors.'},
    ]}]
    assert calls['template']['enable_thinking'] is False
    assert calls['processor']['images'][0].size == (8, 8)
    assert calls['decoded'] == [[41, 42]]
    assert calls['generate']['max_new_tokens'] == 512


def test_zero_temperature_uses_greedy_generation(runtime):
    module, calls, _ = runtime
    assert module.handler(image_job(temperature=0.0))['caption'] == 'A red square.'
    assert calls['generate']['do_sample'] is False
    assert 'temperature' not in calls['generate']
    assert 'top_p' not in calls['generate']


def test_invalid_image_never_reaches_generation(runtime):
    module, calls, _ = runtime
    assert 'error' in module.handler({'input': {'image': 'not an image'}})
    assert 'generate' not in calls


def test_sampling_overrides_reach_generation(runtime):
    module, calls, _ = runtime
    module.handler(image_job(max_new_tokens=128, temperature=0.7, top_p=0.8))
    assert calls['generate']['max_new_tokens'] == 128
    assert calls['generate']['temperature'] == 0.7
    assert calls['generate']['top_p'] == 0.8

