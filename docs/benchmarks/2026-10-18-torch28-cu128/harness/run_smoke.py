import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/opt/torch28-smoke')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=['krea2', 'qwen_image_21'], required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    out = ROOT / 'results'
    out.mkdir(parents=True, exist_ok=True)
    persisted = Path('/workspace/benchmarks/2026-10-18-torch28-cu128') / args.label
    persisted.mkdir(parents=True, exist_ok=True)

    def persist():
        for f in [*out.iterdir(), ROOT / 'status.json']:
            if f.is_file():
                with f.open('rb') as src, (persisted / f.name).open('wb') as dst:
                    shutil.copyfileobj(src, dst)
                    dst.flush()
                    os.fsync(dst.fileno())
                assert hashlib.sha256(f.read_bytes()).digest() == hashlib.sha256((persisted / f.name).read_bytes()).digest()

    def status(**fields):
        fields.update(at=time.time(), worker=args.worker, label=args.label)
        (ROOT / 'status.json').write_text(json.dumps(fields, indent=2))
        print(json.dumps(fields), flush=True)

    def run(command, label):
        with (out / (label + '.log')).open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            status(state='running', stage=label, pid=process.pid)
            try:
                code = process.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = 124
        status(state='stage_finished', stage=label, code=code)
        persist()
        if code:
            raise RuntimeError(f'{label} exited {code}')

    try:
        os.environ.update(HF_HOME='/models/hf', HF_HUB_CACHE='/models/hf/hub',
                          HUGGINGFACE_HUB_CACHE='/models/hf/hub', QWEN_VAE='texture-fix',
                          OFFLOAD='cpu', QWEN_STARTUP_TRACE_INTERVAL='0')
        os.environ.pop('LOCAL_FILES_ONLY', None)
        (out / 'freeze.txt').write_text(subprocess.check_output(['uv', 'pip', 'freeze', '--python', sys.executable], text=True))
        (out / 'nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi', '-q'], text=True))
        for f in [ROOT / 'benchmark.py', ROOT / 'run_smoke.py']:
            shutil.copyfile(f, out / f.name)
        if args.worker == 'krea2':
            os.environ['MODEL_DIR'] = '/models/krea2'
            run([sys.executable, '-u', '/app/download_weights.py', '--output', '/models/krea2',
                 '--hf-home', '/models/hf', '--quant', 'both', '--skip-vae-file'], 'download-krea')
            profiles = ['krea2.int8_convrot', 'krea2.fp8']
            refs = '2'
        else:
            for variant in ['turbo', 'base']:
                run([sys.executable, '-u', '/app/download_weights.py', '--model-variant', variant,
                     '--output', '/models/qwen_image_21_' + variant, '--hf-home', '/models/hf',
                     '--vae-dir', '/models/qwen_image_21_shared/texture_fix_vae'], 'download-' + variant)
            profiles = ['qwen_image_21.turbo', 'qwen_image_21.base']
            refs = '3'
        for profile in profiles:
            run([sys.executable, '-u', str(ROOT / 'benchmark.py'), '--profile', profile,
                 '--refs', refs, '--points', '1', '--label', 'smoke', '--repeats', '2'], profile)
        cases = [json.loads(x) for x in (out / 'events.jsonl').read_text().splitlines()
                 if json.loads(x).get('phase') == 'case']
        assert len(cases) == 4 and all(x['status'] == 'ok' and x['budget_resolved'] == 1 for x in cases)
        status(state='complete', successful_cases=len(cases))
        persist()
    except BaseException as error:
        status(state='failed', error_type=type(error).__name__, error=str(error))
        persist()
        raise


if __name__ == '__main__':
    main()
