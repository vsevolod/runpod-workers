"""Smoke the immutable caption image; do not install or patch its runtime."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

ROOT = Path('/tmp/gliese-smoke')
ROOT.mkdir(exist_ok=True)


def save(name, data):
    (ROOT / name).write_text(json.dumps(data, indent=2))


def main():
    import runpod
    from PIL import Image, ImageDraw
    registered = []
    runpod.serverless.start = lambda config: registered.append(config['handler'])
    sys.path.insert(0, '/app')
    import handler
    assert handler.MODELS is None
    assert 'torch' not in sys.modules, 'Torch imported before downloader startup'
    started = time.monotonic()
    handler.main()
    pipeline_s = time.monotonic() - started
    assert len(registered) == 1
    import torch
    import torchvision
    import transformers
    import accelerate
    assert torch.__version__ == '2.8.0+cu128'
    assert torchvision.__version__ == '0.23.0+cu128'
    assert transformers.__version__ == '5.20.1'
    assert accelerate.__version__ == '1.15.0'
    assert torch.cuda.is_available()
    parameters = list(handler.MODELS.model.parameters())
    assert all(p.device.type == 'cuda' and p.dtype == torch.bfloat16 for p in parameters)
    assert sum(p.numel() for p in parameters) == 9409813744
    torch.ones((32, 32), device='cuda', dtype=torch.bfloat16).matmul(
        torch.ones((32, 32), device='cuda', dtype=torch.bfloat16))
    torch.cuda.synchronize()
    runtime = {
        'python': platform.python_version(), 'torch': torch.__version__,
        'torchvision': torchvision.__version__, 'transformers': transformers.__version__,
        'accelerate': accelerate.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(),
        'capability': list(torch.cuda.get_device_capability()),
        'total_vram_bytes': torch.cuda.get_device_properties(0).total_memory,
        'pipeline_load_s': pipeline_s, 'model': handler.MODELS.model_id,
        'model_revision': handler.MODEL_REVISION,
        'checkpoint_bytes': sum(p.stat().st_size for p in Path(os.environ['MODEL_DIR']).glob('*.safetensors')),
    }
    save('runtime.json', runtime)
    print(json.dumps({'phase': 'ready', **runtime}), flush=True)
    image = Image.new('RGB', (1024, 1024), 'blue')
    ImageDraw.Draw(image).rectangle((256, 256, 768, 768), fill='red')
    image.save(ROOT / 'fixture.png')
    encoded = 'data:image/png;base64,' + base64.b64encode((ROOT / 'fixture.png').read_bytes()).decode()
    cases = []
    for index, (temperature, limit) in enumerate([(0.0, 64), (0.6, 256), (0.6, 512), (0.0, 64)]):
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        result = registered[0]({'id': 'gliese-smoke-' + str(index), 'input': {
            'image': encoded, 'max_new_tokens': limit, 'temperature': temperature,
            'prompt': 'Describe the image in English, including its colors and shapes.'}})
        torch.cuda.synchronize()
        row = {'index': index, 'temperature': temperature, 'max_new_tokens': limit,
               'elapsed_s': time.monotonic() - started, 'result': result,
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
               'max_host_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        cases.append(row)
        save('results.json', {'runtime': runtime, 'fixture_sha256': hashlib.sha256((ROOT / 'fixture.png').read_bytes()).hexdigest(), 'cases': cases})
        print(json.dumps({'phase': 'caption', **row}), flush=True)
        assert not result.get('error'), result
        assert result['model'] == 'prithivMLmods/Gliese-Qwen3.5-9B-Abliterated-Caption'
        assert result['model_revision'] == 'ead05a2ba36c2e68cc03669414210bde75988185'
        assert len(result['caption']) > 20
        assert 'red' in result['caption'].lower() and 'blue' in result['caption'].lower()
    assert cases[0]['result']['caption'] == cases[3]['result']['caption']
    save('status.json', {'state': 'complete', 'successful_cases': len(cases)})
    print('GLIESE_SMOKE_OK', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        save('status.json', {'state': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
        raise
