"""Summarize observed MOSS capacity without extrapolating across GPUs or inputs."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
protocol = json.loads((ROOT / 'protocol.json').read_text())
rows, hardware, confirmations, prompts = [], {}, [], set()
for folder in sorted((ROOT / 'results').iterdir()):
    if not folder.is_dir():
        continue
    events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
    loads = [r for r in events if r.get('phase') == 'loading']
    assert loads and all(r['versions']['torch'] == '2.9.1+cu128' for r in loads)
    hardware[folder.name] = loads[0]
    summaries = json.loads((folder / 'summary.json').read_text())
    cases = [r for r in events if r.get('phase') == 'case']
    global_hashes = json.loads((folder / 'global-sha256.json').read_text())
    assert json.loads((folder / 'status.json').read_text())['state'] == 'complete'
    prompts.update(r['input']['prompt'] for r in cases)
    for summary in summaries:
        side = summary['confirmed_side_with_headroom']
        assert side, (folder.name, summary)
        checked = [r for r in cases if r['profile'] == 'moss_image.' + summary['variant']
                   and r['label'] == f'confirm-{side}']
        assert len(checked) == 2 and all(r['status'] == 'ok' for r in checked)
        assert [r['repeat'] for r in checked] == [0, 1]
        assert any(r['profile'] == checked[0]['profile'] and r['label'] == checked[0]['label'] for r in loads)
        sweep = [r for r in cases if r['profile'] == checked[0]['profile'] and r['label'] == 'sweep']
        assert [r['width'] for r in sweep] == protocol['square_sides'][:len(sweep)]
        if len(sweep) < len(protocol['square_sides']):
            assert sweep[-1]['status'] == 'oom'
        assert all(r['status'] == 'ok' for r in sweep[:-1])
        assert all(r['conservative_headroom_bytes'] >= max(1024**3, r['total_vram_bytes'] * .05) for r in checked)
        confirmations.append({'hardware': folder.name, **summary,
                              'max_confirmed_device_gib_sampled': max(r['peak_device_used_bytes_sampled'] for r in checked) / 2**30,
                              'max_confirmed_conservative_gib': max(r['conservative_device_peak_bytes'] for r in checked) / 2**30,
                              'min_confirmed_headroom_gib': min(r['conservative_headroom_bytes'] for r in checked) / 2**30,
                              'cold_seconds': checked[0]['inference_seconds'],
                              'warm_seconds': checked[1]['inference_seconds'],
                              'cold_warm_identical_png': checked[0]['image_sha256'] == checked[1]['image_sha256']})
    for r in cases:
        if r['status'] == 'ok':
            assert global_hashes[r['image']] == r['image_sha256']
            assert r['model_metadata']['model_revision'] == protocol['model_revision']
            assert all(s[-2:] == [r['height'] // 8, r['width'] // 8] for s in r['vae_decode_shapes'])
            assert r['vae_decode_shapes'] and r['transformer_shapes']
            assert all(s[-2:] == [r['height'] // 16, r['width'] // 16] for s in r['transformer_shapes'])
            if r['profile'].endswith('.turbo'):
                assert r['model_metadata']['lora_revision'] == protocol['lora_revision']
                assert len(r['loras']) == 1 and r['loras'][0]['strength'] == protocol['lora_strength']
            else:
                assert r['loras'] == [] and r['model_metadata']['lora_revision'] is None
        rows.append({'hardware': folder.name, 'profile': r['profile'], 'label': r['label'],
                     'repeat': r['repeat'], 'status': r['status'], 'width': r['width'], 'height': r['height'],
                     'output_mp': r['megapixels'], 'steps': r['steps'], 'cfg': r['guidance'],
                     'allocated_gib': r['peak_allocated_bytes'] / 2**30,
                     'reserved_gib': r['peak_reserved_bytes'] / 2**30,
                     'device_gib_sampled': r['peak_device_used_bytes_sampled'] / 2**30,
                     'conservative_device_gib': r['conservative_device_peak_bytes'] / 2**30,
                     'conservative_headroom_gib': r['conservative_headroom_bytes'] / 2**30,
                     'host_max_rss_gib': r['max_rss_kib'] / 2**20,
                     'cgroup_peak_gib_sampled': r['peak_cgroup_bytes_sampled'] / 2**30
                     if r['peak_cgroup_bytes_sampled'] else None,
                     'seconds': r['inference_seconds'], 'image_sha256': r.get('image_sha256'),
                     'error': r.get('error')})
assert rows
assert len(prompts) == 1, 'Keep the generation prompt fixed throughout the capacity sweep'
with (ROOT / 'measurements.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
report = {'protocol': protocol, 'hardware': hardware,
          'prompt_sha256': hashlib.sha256(next(iter(prompts)).encode()).hexdigest(),
          'successful_requests': sum(r['status'] == 'ok' for r in rows),
          'oom_requests': sum(r['status'] == 'oom' for r in rows),
          'confirmations': confirmations, 'rows': rows}
(ROOT / 'measurements.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ['successful_requests', 'oom_requests', 'confirmations']}, indent=2))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
for name in hardware:
    for variant in ['base', 'turbo']:
        values = [r for r in rows if r['hardware'] == name and r['profile'] == 'moss_image.' + variant
                  and r['label'] == 'sweep' and r['status'] == 'ok']
        label = name + ' / ' + variant
        memory_line, = axes[0].plot([r['output_mp'] for r in values], [r['device_gib_sampled'] for r in values], marker='o', label=label)
        time_line, = axes[1].plot([r['output_mp'] for r in values], [r['seconds'] for r in values], marker='o', label=label)
        confirmed = [r for r in rows if r['hardware'] == name and r['profile'] == 'moss_image.' + variant
                     and r['label'].startswith('confirm-') and r['status'] == 'ok']
        axes[0].scatter([r['output_mp'] for r in confirmed], [r['device_gib_sampled'] for r in confirmed],
                        color=memory_line.get_color(), marker='x', s=55, zorder=4)
        axes[1].scatter([r['output_mp'] for r in confirmed], [r['seconds'] for r in confirmed],
                        color=time_line.get_color(), marker='x', s=55, zorder=4)
axes[0].set(ylabel='Sampled whole-device peak (GiB)', title='GPU memory (100 ms sampling)')
for index, (name, info) in enumerate(hardware.items()):
    axes[0].axhline(info['total_vram_bytes'] / 2**30, color='gray',
                   ls='--' if index == 0 else ':', label=name + ' CUDA total')
axes[0].set_ylim(0, max(info['total_vram_bytes'] for info in hardware.values()) / 2**30 + 2)
axes[1].set(ylabel='Inference plus adapter attachment (seconds)', title='Generation time')
for ax in axes:
    ax.scatter([], [], color='black', marker='x', label='Fresh/warm maximum repeats')
    ax.set_xlabel('Actual output megapixels')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle('MOSS SFT / DMD 8-step — pinned deployed image, CPU offload, one square output')
fig.savefig(ROOT / 'vram-and-time.png', dpi=170)
fig.savefig(ROOT / 'vram-and-time.pdf')
