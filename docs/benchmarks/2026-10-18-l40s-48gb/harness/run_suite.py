import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/opt/bench')
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
PERSIST = Path('/workspace/benchmarks/2026-10-18-L40S-48GB')
PERSIST.mkdir(parents=True, exist_ok=True)


def persist():
    for path in OUT.iterdir():
        if path.is_file():
            with path.open('rb') as src, (PERSIST / path.name).open('wb') as dst:
                shutil.copyfileobj(src, dst)
                dst.flush()
                os.fsync(dst.fileno())


def status(**fields):
    fields['at'] = time.time()
    (ROOT / 'status.json').write_text(json.dumps(fields, indent=2))
    print(json.dumps(fields), flush=True)


def run(profile, refs, points, label, repeats=1):
    python = '/opt/venvs/' + ('krea' if profile.startswith('krea2') else 'qwen') + '/bin/python'
    command = [python, '-u', str(ROOT / 'benchmark.py'), '--profile', profile,
               '--refs', str(refs), '--points', points, '--label', label, '--repeats', str(repeats)]
    with (ROOT / f'{profile}-{label}.log').open('w') as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        status(state='running', profile=profile, refs=refs, label=label, pid=process.pid)
        try:
            code = process.wait(timeout=5400)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            code = 124
    status(state='profile_finished', profile=profile, label=label, code=code)
    persist()
    return code


def main():
    profiles = [('krea2.int8_convrot', 2), ('krea2.fp8', 2),
                ('qwen_image_21.turbo', 3), ('qwen_image_21.base', 3)]
    summaries = []
    for profile, refs in profiles:
        events = [json.loads(line) for line in (OUT / 'events.jsonl').read_text().splitlines()]
        ready = any(e.get('profile') == profile and e.get('phase') == 'case'
                    and e.get('status') == 'ok' and e.get('mp_requested') == 1 for e in events)
        if not ready:
            run(profile, refs, '1', 'preflight')
    for profile, refs in profiles:
        code = run(profile, refs, '1.5,2,2.5,3,3.5,4', 'sweep')
        events = [json.loads(line) for line in (OUT / 'events.jsonl').read_text().splitlines()]
        passed = [e for e in events if e.get('profile') == profile and e.get('phase') == 'case' and e.get('status') == 'ok']
        best = max((e['mp_requested'] for e in passed), default=None)
        confirm_code = None
        if best is not None and code in (0, 22):
            confirm_code = run(profile, refs, str(best), 'confirm', repeats=2)
        summaries.append({'profile': profile, 'refs': refs, 'sweep_code': code,
                          'highest_successful_mp': best, 'confirmation_code': confirm_code})
        (OUT / 'summary.json').write_text(json.dumps(summaries, indent=2))
        persist()
    status(state='complete', profiles=summaries)
    persist()


if __name__ == '__main__':
    main()
