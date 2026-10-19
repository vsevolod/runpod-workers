"""Sequential subprocesses, fresh maximum confirmation, durable Global results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/opt/moss-bench')
OUT = ROOT / 'results'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    persist_root = Path('/workspace/benchmarks/2026-10-19-moss') / args.label
    persist_root.mkdir(parents=True, exist_ok=True)
    assert Path('/workspace').is_mount(), 'Global volume must be mounted before starting'
    os.environ.update(MOSS_MODEL_DIR='/models/moss_image_sft', MOSS_OFFLOAD='cpu',
                      HF_HOME='/models/hf', TOKENIZERS_PARALLELISM='false')

    def persist():
        manifest = {}
        for path in [*OUT.iterdir(), ROOT / 'status.json']:
            if not path.is_file():
                continue
            # Logs may still be growing; copy a fixed snapshot before hashing.
            data = path.read_bytes()
            target = persist_root / path.name
            with target.open('wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            digest = hashlib.sha256(data).hexdigest()
            assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
            manifest[path.name] = digest
        with (persist_root / 'sha256.json').open('w') as stream:
            json.dump(manifest, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())

    def status(**fields):
        fields.update(at=time.time(), label=args.label)
        (ROOT / 'status.json').write_text(json.dumps(fields, indent=2))
        print(json.dumps(fields), flush=True)

    def run(command, stage, timeout=5400):
        with (OUT / (stage + '.log')).open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            status(state='running', stage=stage, pid=process.pid)
            started = time.monotonic()
            while process.poll() is None:
                persist()
                if time.monotonic() - started > timeout:
                    process.kill()
                    process.wait()
                    code = 124
                    break
                time.sleep(10)
            else:
                code = process.returncode
        status(state='stage_finished', stage=stage, code=code)
        persist()
        return code

    def cases():
        path = OUT / 'events.jsonl'
        return [record for line in path.read_text().splitlines()
                if (record := json.loads(line)).get('phase') == 'case'] if path.exists() else []

    try:
        (OUT / 'freeze.txt').write_text(subprocess.check_output(['uv', 'pip', 'freeze', '--python', sys.executable], text=True))
        (OUT / 'nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi', '-q'], text=True))
        for path in [ROOT / 'benchmark.py', ROOT / 'run_suite.py', ROOT / 'protocol.json']:
            shutil.copyfile(path, OUT / path.name)
        code = run([sys.executable, '-u', '/app/workers/moss_image/download_weights.py',
                    '--turbo', '--output', os.environ['MOSS_MODEL_DIR'], '--hf-home', '/models/hf'], 'download')
        if code:
            raise RuntimeError(f'download failed: {code}')
        for relative in ['.moss-model.json', 'turbo/.moss-lora.json']:
            shutil.copyfile(Path(os.environ['MOSS_MODEL_DIR']) / relative, OUT / Path(relative).name)
        summaries = []
        for variant in ['base', 'turbo']:
            command = [sys.executable, '-u', str(ROOT / 'benchmark.py'), '--variant', variant]
            code = run(command, variant + '-sweep')
            if code not in (0, 22):
                raise RuntimeError(f'{variant} sweep failed: {code}')
            passed = [r for r in cases() if r['profile'] == 'moss_image.' + variant and r['status'] == 'ok']
            candidates = sorted({r['width'] for r in passed
                                 if r['conservative_headroom_bytes'] >= max(1024**3, r['total_vram_bytes'] * .05)}, reverse=True)
            confirmed = None
            for side in candidates:
                label = f'confirm-{side}'
                code = run(command + ['--sides', str(side), '--repeats', '2', '--label', label], variant + '-' + label)
                checked = [r for r in cases() if r['profile'] == 'moss_image.' + variant and r['label'] == label]
                if code == 0 and len(checked) == 2 and all(
                        r['status'] == 'ok' and r['conservative_headroom_bytes'] >= max(1024**3, r['total_vram_bytes'] * .05)
                        for r in checked):
                    confirmed = side
                    break
                if code not in (0, 22):
                    raise RuntimeError(f'{variant} confirmation failed: {code}')
            summaries.append({'variant': variant, 'highest_successful_side': max((r['width'] for r in passed), default=None),
                              'confirmed_side_with_headroom': confirmed,
                              'confirmed_megapixels': confirmed**2 / 1_000_000 if confirmed else None})
            (OUT / 'summary.json').write_text(json.dumps(summaries, indent=2))
            persist()
        status(state='complete', summaries=summaries)
        persist()
    except BaseException as error:
        status(state='failed', error_type=type(error).__name__, error=str(error))
        persist()
        raise


if __name__ == '__main__':
    main()
