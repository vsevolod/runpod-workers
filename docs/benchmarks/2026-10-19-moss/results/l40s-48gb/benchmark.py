"""Adapt the preserved capacity harness to the unchanged deployed MOSS runtime."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import resource
import sys
import threading
import time
import traceback

ROOT = Path('/opt/moss-bench')
OUT = ROOT / 'results'
OUT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')


def emit(record):
    record['at'] = time.time()
    with (OUT / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    print('BENCH_EVENT ' + json.dumps(record), flush=True)


def cgroup_value(name):
    for root in ['/sys/fs/cgroup', '/sys/fs/cgroup/memory']:
        try:
            value = (Path(root) / name).read_text().strip()
            return int(value) if value != 'max' else value
        except (OSError, ValueError):
            pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['base', 'turbo'], required=True)
    parser.add_argument('--sides', default='992,1216,1408,1568,1728,1856,2000,2048')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--label', default='sweep')
    args = parser.parse_args()
    import torch
    import torchvision
    from PIL import ImageStat
    from shared.api import MOSS_API, parse_job_input
    from moss_image_infer.config import MODEL_ID, MODEL_REVISION, LORA_ID, LORA_REVISION, LORA_STRENGTH
    from moss_image_infer.pipeline import load_pipeline
    from moss_image_infer.request import normalize_job_input

    assert torch.__version__ == '2.9.1+cu128', torch.__version__
    assert torchvision.__version__ == '0.24.1+cu128', torchvision.__version__
    assert torch.version.cuda == '12.8' and torch.cuda.is_available()
    torch.set_num_threads(8)
    assert (torch.ones(1, device='cuda') * 2).item() == 2
    props = torch.cuda.get_device_properties(0)
    profile = 'moss_image.' + args.variant
    versions = {p: importlib.metadata.version(p) for p in [
        'torch', 'torchvision', 'diffusers', 'transformers', 'accelerate', 'huggingface-hub', 'safetensors']}
    emit({'phase': 'loading', 'profile': profile, 'label': args.label, 'gpu': props.name,
          'total_vram_bytes': props.total_memory, 'versions': versions,
          'cuda_build': torch.version.cuda, 'compute_capability': list(torch.cuda.get_device_capability()),
          'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
          'lora_id': LORA_ID if args.variant == 'turbo' else None,
          'lora_revision': LORA_REVISION if args.variant == 'turbo' else None,
          'lora_strength': LORA_STRENGTH if args.variant == 'turbo' else 0,
          'offload': os.environ['MOSS_OFFLOAD'], 'cgroup_limit_bytes': cgroup_value('memory.max')})
    started = time.perf_counter()
    runtime = load_pipeline(os.environ['MOSS_MODEL_DIR'])
    torch.cuda.synchronize()
    emit({'phase': 'ready', 'profile': profile, 'label': args.label,
          'load_seconds': time.perf_counter() - started})
    decode_shapes, transformer_shapes = [], []
    original_decode = runtime.pipe.vae.decode

    def traced_decode(pixels, *positional, **kwargs):
        decode_shapes.append(list(pixels.shape))
        return original_decode(pixels, *positional, **kwargs)

    runtime.pipe.vae.decode = traced_decode

    def trace_transformer(module, inputs):
        shape = list(inputs[0].shape)
        if shape not in transformer_shapes:
            transformer_shapes.append(shape)

    handle = runtime.pipe.transformer.register_forward_pre_hook(trace_transformer)
    for side in map(int, args.sides.split(',')):
        for repeat in range(args.repeats):
            inputs = {'type': 'image_generate',
                      'prompt': 'A studio product photograph of a red ceramic teapot and exactly one blue cup on a light wooden table. A white card beside them clearly reads MOSS. Soft window light, realistic materials, uncluttered background.',
                      'width': side, 'height': side, 'seed': 42, 'num_images': 1,
                      'model_options': {'variant': args.variant}}
            parsed = parse_job_input({'input': inputs}, MOSS_API)
            norm = normalize_job_input(parsed.values, raw_keys=parsed.raw_keys)
            assert norm.width == side and norm.height == side and norm.num_images == 1
            record = {'phase': 'case', 'label': args.label, 'profile': profile, 'repeat': repeat,
                      'megapixels': side * side / 1_000_000, 'width': side, 'height': side,
                      'seed': norm.seed, 'steps': norm.num_inference_steps, 'guidance': norm.guidance_scale,
                      'shift': norm.shift, 'uncond_skip_layers': list(norm.uncond_skip_layers), 'input': inputs}
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            decode_shapes.clear()
            transformer_shapes.clear()
            free, total = torch.cuda.mem_get_info()
            nonallocator_baseline = max(0, total - free - torch.cuda.memory_reserved())
            stop = threading.Event()
            samples, host_samples = [], []

            def sample_memory():
                while not stop.is_set():
                    free, total = torch.cuda.mem_get_info()
                    samples.append(total - free)
                    host = cgroup_value('memory.current')
                    if isinstance(host, int):
                        host_samples.append(host)
                    stop.wait(.1)

            watcher = threading.Thread(target=sample_memory, daemon=True)
            watcher.start()
            emit({**record, 'phase': 'case_begin'})
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    result = runtime.run(norm, seed=42)
                torch.cuda.synchronize()
                record.update(status='ok', inference_seconds=time.perf_counter() - started,
                              model_metadata=result.metadata,
                              loras=[selection.as_dict() for selection in result.loras])
                assert len(result.images) == 1 and result.images[0].size == (side, side)
                record['channel_stddev'] = ImageStat.Stat(result.images[0].convert('RGB')).stddev
                assert max(record['channel_stddev']) > 1, 'Blank output'
            except Exception as error:
                record.update(status='oom' if isinstance(error, torch.cuda.OutOfMemoryError) else 'error',
                              inference_seconds=time.perf_counter() - started,
                              error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            finally:
                stop.set()
                watcher.join(timeout=5)
                reserved = torch.cuda.max_memory_reserved()
                sampled = max(samples, default=0)
                conservative = max(sampled, reserved + nonallocator_baseline)
                record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=reserved, peak_device_used_bytes_sampled=sampled,
                              device_samples=len(samples), sample_interval_seconds=.1,
                              nonallocator_baseline_bytes=nonallocator_baseline,
                              conservative_device_peak_bytes=conservative,
                              conservative_headroom_bytes=props.total_memory - conservative,
                              total_vram_bytes=props.total_memory,
                              vae_decode_shapes=decode_shapes.copy(),
                              transformer_shapes=transformer_shapes.copy(),
                              max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                              peak_cgroup_bytes_sampled=max(host_samples, default=0))
            if record['status'] == 'ok':
                name = f'{profile}-{args.label}-{side}-r{repeat}.png'
                result.images[0].save(OUT / name)
                record.update(image=name, image_sha256=hashlib.sha256((OUT / name).read_bytes()).hexdigest())
                del result
            emit(record)
            if record['status'] != 'ok':
                return 22 if record['status'] == 'oom' else 1
    handle.remove()
    runtime.close()
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        emit({'phase': 'startup_error', 'error_type': type(error).__name__, 'error': str(error),
              'traceback': traceback.format_exc()})
        raise
