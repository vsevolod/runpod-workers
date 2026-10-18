"""Measure unchanged worker inference paths on one dedicated GPU."""
import argparse
import base64
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import resource
import sys
import threading
import time
import traceback

REPO = Path('/app')
OUT = Path('/opt/torch28-smoke/results')
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')


def emit(record):
    record['at'] = time.time()
    with (OUT / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
        stream.flush()
    print('BENCH_EVENT ' + json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True)
    parser.add_argument('--points', default='1,1.5,2,2.5,3,3.5,4')
    parser.add_argument('--refs', type=int, required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--label', default='sweep')
    args = parser.parse_args()
    is_krea = args.profile.startswith('krea2.')
    variant = args.profile.split('.')[1]
    os.environ.update(HF_HOME='/models/hf', LOCAL_FILES_ONLY='1',
                      HF_HUB_CACHE='/models/hf/hub', HUGGINGFACE_HUB_CACHE='/models/hf/hub',
                      OFFLOAD='cpu', QWEN_VAE='texture-fix', QWEN_STARTUP_TRACE_INTERVAL='0')
    if is_krea:
        os.environ.update(MODEL_DIR='/models/krea2', DIT_QUANT=variant,
                          LORA_DIR='/workspace/krea2/loras')
        sys.path.insert(0, str(REPO))
    else:
        os.environ.update(MODEL_DIR='/models/qwen_image_21_turbo',
                          BASE_MODEL_DIR='/models/qwen_image_21_base',
                          QWEN_VAE_DIR='/models/qwen_image_21_shared/texture_fix_vae',
                          LORA_DIR='/workspace/qwen_image_21_turbo/loras')
        sys.path.insert(0, str(REPO))
    import torch
    import torchvision
    assert torch.__version__ == '2.8.0+cu128', torch.__version__
    assert torchvision.__version__ == '0.23.0+cu128', torchvision.__version__
    assert torch.version.cuda == '12.8', torch.version.cuda
    from PIL import ImageStat
    from shared.api import KREA_API, QWEN_API, parse_job_input
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    props = torch.cuda.get_device_properties(0)
    versions = {}
    for package in ['torch', 'torchvision', 'diffusers', 'transformers', 'peft', 'accelerate', 'huggingface-hub']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    emit({'phase': 'loading', 'profile': args.profile, 'gpu': props.name,
          'total_vram_bytes': props.total_memory, 'versions': versions,
          'cuda_build': torch.version.cuda, 'python': platform.python_version(), 'compute_capability': list(torch.cuda.get_device_capability(0))})
    started = time.perf_counter()
    if is_krea:
        from krea2_infer import load_pipeline
        from krea2_infer.request import normalize_job_input
        runtime = load_pipeline()
        normalize = lambda values, raw_keys: normalize_job_input(values, raw_keys=raw_keys, quant_mode=variant)
        lora_name = 'krea2_identity_edit_v2'
        vae = runtime.ae
    else:
        from qwen_image_21_infer.pipeline import load_pipeline
        from qwen_image_21_infer.request import normalize_job_input
        runtime = load_pipeline(model_variant=variant)
        normalize = lambda values, raw_keys: normalize_job_input(values, raw_keys=raw_keys)
        lora_name = 'lenovo_qwen21'
        vae = runtime.pipe.vae
    torch.cuda.synchronize()
    emit({'phase': 'ready', 'profile': args.profile, 'load_seconds': time.perf_counter() - started,
          'lora': lora_name, 'lora_source': str(runtime.loras.catalog.resolve(lora_name)) if is_krea
          else str(runtime.loras.catalog._paths[lora_name])})
    vae_shapes = []
    original_encode = vae.encode
    def traced_encode(pixels, *positional, **kwargs):
        vae_shapes.append(list(pixels.shape))
        return original_encode(pixels, *positional, **kwargs)
    vae.encode = traced_encode
    images = []
    for index in range(args.refs):
        data = (Path('/opt/torch28-smoke/inputs') / f'ref-{index}.jpg').read_bytes()
        images.append('data:image/jpeg;base64,' + base64.b64encode(data).decode())
    for mp in map(float, args.points.split(',')):
        repetitions = args.repeats if args.label != 'sweep' or mp != 1 else max(2, args.repeats)
        for repeat in range(repetitions):
            inputs = {'type': 'image_edit', 'prompt': 'Change the jacket to a blue jacket. Preserve the person, face, pose, composition and natural lighting.',
                      'images': images, 'seed': 42, 'reference_megapixels': 'max',
                      'loras': [{'name': lora_name, 'strength': 1.0}],
                      'guidance_scale': 0.0 if is_krea else 1.0,
                      'num_inference_steps': 40 if args.profile == 'qwen_image_21.base' else 8}
            if not is_krea:
                inputs['model_options'] = {'variant': variant}
            parsed = parse_job_input({'input': inputs}, KREA_API if is_krea else QWEN_API)
            norm = normalize(parsed.values, parsed.raw_keys)
            assert norm.reference_megapixels == 1., norm.reference_megapixels
            selections = runtime.loras.normalize(inputs['loras'])
            record = {'phase': 'case', 'label': args.label, 'profile': args.profile, 'mp_requested': mp,
                      'refs': args.refs, 'repeat': repeat, 'lora': lora_name, 'lora_strength': 1.0,
                      'seed': 42, 'budget_requested': 'max', 'budget_resolved': norm.reference_megapixels, 'steps': norm.num_inference_steps, 'guidance': norm.guidance_scale,
                      'width': norm.width, 'height': norm.height,
                      'reference_sizes_after_budget': [list(image.size) for image in norm.images]}
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            vae_shapes.clear()
            stop = threading.Event()
            samples = []
            def sample_memory():
                while not stop.is_set():
                    free, total = torch.cuda.mem_get_info()
                    samples.append(total - free)
                    stop.wait(.1)
            watcher = threading.Thread(target=sample_memory, daemon=True)
            watcher.start()
            emit({**record, 'phase': 'case_begin'})
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    if is_krea:
                        result = runtime.edit(prompt=norm.prompt, sources=norm.images, width=norm.width,
                            height=norm.height, steps=norm.num_inference_steps, guidance=norm.guidance_scale,
                            seed=42, mu=norm.mu, negative_prompt=norm.negative_prompt, loras=selections,
                            grounding_px=norm.grounding_px, ref_boost=norm.ref_boost,
                            ref_boost_a=norm.ref_boost_a, ref_position=norm.ref_position,
                            ref_position_a=norm.ref_position_a, fit_mode=norm.fit_mode)
                    else:
                        result = runtime.run(norm, seed=42, loras=selections)
                torch.cuda.synchronize()
                record.update(status='ok', inference_seconds=time.perf_counter() - started)
                assert len(result) == 1 and result[0].size == (norm.width, norm.height)
                record['channel_stddev'] = ImageStat.Stat(result[0].convert('RGB')).stddev
                assert max(record['channel_stddev']) > 1, 'Blank output'
                name = f'{args.profile}-{args.label}-refs{args.refs}-mp{mp:g}-r{repeat}.png'
                result[0].save(OUT / name)
                record['image'] = name
                del result
            except Exception as error:
                record.update(status='oom' if isinstance(error, torch.cuda.OutOfMemoryError) else 'error',
                              inference_seconds=time.perf_counter() - started,
                              error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            finally:
                stop.set()
                watcher.join(timeout=5)
                record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                              peak_device_used_bytes_sampled=max(samples, default=0),
                              total_vram_bytes=props.total_memory, vae_encode_shapes=vae_shapes.copy(),
                              max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                emit(record)
            if record['status'] != 'ok':
                return 22 if record['status'] == 'oom' else 1
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        emit({'phase': 'startup_error', 'error_type': type(error).__name__, 'error': str(error),
              'traceback': traceback.format_exc()})
        raise
