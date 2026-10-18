"""Export observed GPU benchmark curves; no extrapolation to other hardware."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics

ROOT = Path(__file__).parent
events = [json.loads(line) for line in (ROOT / 'results/events.jsonl').read_text().splitlines()]
cases = [event for event in events if event.get('phase') == 'case']
assert cases and all(case['status'] == 'ok' for case in cases), 'Generation failures require review'
profiles = ['krea2.int8_convrot', 'krea2.fp8', 'qwen_image_21.turbo', 'qwen_image_21.base']
points = [1., 1.5, 2., 2.5, 3., 3.5, 4.]
groups = defaultdict(list)
for case in cases:
    groups[(case['profile'], case['mp_requested'])].append(case)
rows = []
for profile in profiles:
    for mp in points:
        runs = groups[(profile, mp)]
        assert runs, (profile, mp)
        if mp == 4:
            assert len([run for run in runs if run['label'] == 'confirm']) >= 2, profile
        first = runs[0]
        rows.append(dict(profile=profile, mp_requested=mp, width=first['width'], height=first['height'],
            output_mp=first['width'] * first['height'] / 1e6, refs=first['refs'],
            lora=first['lora'], strength=first['lora_strength'], steps=first['steps'],
            runs=len(runs), max_device_peak_gib=max(r['peak_device_used_bytes_sampled'] for r in runs) / 2**30,
            max_allocated_gib=max(r['peak_allocated_bytes'] for r in runs) / 2**30,
            max_reserved_gib=max(r['peak_reserved_bytes'] for r in runs) / 2**30,
            min_seconds=min(r['inference_seconds'] for r in runs),
            mean_seconds=statistics.mean(r['inference_seconds'] for r in runs),
            max_seconds=max(r['inference_seconds'] for r in runs)))
with (ROOT / 'measurements.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
report = dict(successful_requests=len(cases), gpu='NVIDIA L40S', nominal_gpu_gb=48,
    cuda_available_gib=cases[0]['total_vram_bytes'] / 2**30,
    tested_mp_ceiling=4, baseline='Krea includes Identity Edit V2; Qwen includes Lenovo adapter',
    device_peak_sampling_interval_seconds=.1,
    configurations='Square edits; Krea two refs / 8 steps / CFG 0; Qwen three refs / Turbo 8 or Base 40 steps / CFG 1; one output',
    scope='Observed results on this GPU and pinned test environment; other GPU capacities and extra adapters untested',
    rows=rows)
(ROOT / 'measurements.json').write_text(json.dumps(report, indent=2))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
labels = {'krea2.int8_convrot': 'Krea INT8, 8 steps', 'krea2.fp8': 'Krea FP8, 8 steps',
          'qwen_image_21.turbo': 'Qwen Turbo, 8 steps', 'qwen_image_21.base': 'Qwen Base, 40 steps'}
for profile in profiles:
    values = [row for row in rows if row['profile'] == profile]
    axes[0].plot(points, [row['max_device_peak_gib'] for row in values], marker='o', label=labels[profile])
    axes[1].plot(points, [row['mean_seconds'] for row in values], marker='o', label=labels[profile])
axes[0].axhline(report['cuda_available_gib'], color='gray', ls='--', label='Available CUDA memory')
axes[0].set(ylabel='Observed device peak (GiB)', title='GPU memory', ylim=(0, report['cuda_available_gib'] + 2))
axes[1].set(ylabel='Inference time (seconds)', title='Runtime')
for ax in axes:
    ax.set_xlabel('Requested megapixels')
    ax.set_xticks(points)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle('L40S 48 GB — fixed LoRA baselines, square edits')
fig.savefig(ROOT / 'vram-and-time.png', dpi=170)
fig.savefig(ROOT / 'vram-and-time.pdf')
print(json.dumps({key: report[key] for key in ['successful_requests', 'cuda_available_gib', 'tested_mp_ceiling']}))
for row in rows:
    if row['mp_requested'] == 4:
        print(row['profile'], round(row['max_device_peak_gib'], 2), round(row['mean_seconds'], 1), 'seconds')
