# MOSS Image capacity — 2026-10-19

This run measures the deployed native MOSS backend through its real request
parser, normalization and inference. It uses dedicated GPU Pods, explicit output
dimensions and no reference images. It measures capacity on a fixed generation
fixture; it is not an image-quality comparison.

## Confirmed results

**40/40 successful generations; zero CUDA OOMs.** Base and Turbo each
passed all eight square sizes on both GPUs, then confirmed **2048x2048
(4.194304 MP)** twice in a fresh process with the required memory margin.
All PNG dimensions and nonblank checks passed. All 40 output PNGs and the
other run files were SHA-256 checked against Global storage before teardown.
The first and warm maximum-size PNGs matched byte-for-byte for all four
GPU/profile combinations. The four maximum-size inspection images were
visually checked; this is not a model-quality evaluation.

| GPU | Profile | Peak device use at 2048 squared (GiB) | Minimum headroom (GiB) | First fresh request (s) | Warm repeat (s) |
|---|---|---:|---:|---:|---:|
| RTX 4090 / 24 GB | base | 16.15 | 7.37 | 278.9 | 278.1 |
| RTX 4090 / 24 GB | turbo | 16.52 | 6.99 | 50.1 | 48.8 |
| L40S / 48 GB | base | 16.19 | 28.20 | 347.4 | 353.2 |
| L40S / 48 GB | turbo | 16.19 | 28.20 | 49.7 | 49.9 |

Peak/headroom take the worst of the sweep and two confirmation requests.
Times above are the fresh-process confirmation pair; their timing boundaries
are defined below. GiB are binary; GPU tier names are nominal GB.

| Hardware | Actual CUDA VRAM (GiB) | Allocated host RAM (GB) | Largest process RSS (GiB) | Largest sampled container memory (GiB) |
|---|---:|---:|---:|---:|
| rtx4090-24gb | 23.516 | 100 | 33.88 | N/A |
| l40s-48gb | 44.392 | 125 | 34.03 | 59.16 |

The queried cgroup memory counters were unavailable on the RTX 4090 Pod;
raw events use zero as the empty-sample sentinel, and derived CSV/JSON values
use missing/null. This does not mean zero host RAM usage: process RSS was
measured on both GPUs. Container RAM is separate from GPU VRAM.

Both dedicated Pods were stopped and deleted after durable results and local
inspection copies were secured. Final API reads returned HTTP 404 for both
IDs; see [`resources.json`](resources.json). The production endpoint retained
the deployed image and its 0..1 workers / 10s idle configuration; the
[sanitized readback](endpoint-readback.json) records this separately.

The confirmed scope is single-image square generation with CPU offload and
this short fixed prompt. It supports the existing API ceiling in this
configuration; it does not establish a universal VRAM-dependent output policy.

## Pinned baseline

- Image: `ghcr.io/vsevolod/runpod-workers-image@sha256:fbe62a59c6fbc9099c1f48fa49529eddb34446ecd7b52cb0830f1850ab35b7a9`.
- Worker source: `d95b30908773c9d0d5d49ad69ede571ad1caffe5`.
  Source hashes are checked inside each container before starting and preserved
  in each hardware folder's `source-verification.json`.
- Torch `2.9.1+cu128`, torchvision `0.24.1+cu128`; installed package versions,
  freezes and initial `nvidia-smi -q` reports are preserved with the measurements.
- `OpenMOSS-Team/MOSS-Image-SFT` at
  `2c7954437269461767f533b006823e8f85973a60`.
- Base is SFT without an accelerator: 50 steps, CFG 3, negative-branch skip
  layer `[22]`. It does not select the original MOSS-Image-Base checkpoint.
- Turbo uses `lodestones/MOSS-Image-SFT-8-steps-dmd` at
  `f95df83b5e90ae91ea412434b3f9b7f1c8c6314f`, `lora_latest.safetensors`
  unmerged at strength 0.7: eight steps, CFG 1, no negative-branch skipping.
- Component CPU offload (`MOSS_OFFLOAD=cpu`), BF16 transformer/text encoder,
  FP32 FLUX.2 small VAE/noise/Euler updates. Euler shift 5, seed 42, one output,
  square aspect ratio, one fixed short English prompt. Eight Torch CPU threads.
- Image defaults, including `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  and `TORCH_COMPILE_DISABLE=1`, are inherited from the pinned image.
- Square sides: 992, 1216, 1408, 1568, 1728, 1856, 2000, 2048. The report uses
  actual decimal MP, rather than rounding aligned dimensions to nominal targets.
  2000 squared is exactly 4 MP; 2048 squared is 4.194304 MP, the API area ceiling.

The complete configuration and inference-script hashes are in
[`protocol.json`](protocol.json). Weights and caches stay on container disk;
Global storage receives results and logs, not another copy of the checkpoints.

## Measurement and confirmation

Each GPU runs Base and Turbo sequentially. Each profile sweeps upward until
the API ceiling or its first CUDA OOM, then exits that process. A fresh process
loads the model and generates the highest successful candidate twice: its first
request and one warm repeat. Candidates must retain at least 1 GiB or 5% of
actual CUDA VRAM, whichever is larger. An insufficient margin or OOM during
confirmation selects a smaller previously successful candidate in another
fresh process.

Measurements include:

- CUDA allocator peak allocated/reserved bytes and sampled whole-device use
  (`mem_get_info`, 100 ms intervals). Samples can miss short spikes. The
  headroom estimate takes the larger of sampled device use and peak reserved
  plus the nonallocator use measured before inference.
- Actual CUDA total memory, GPU identity, host process maximum RSS and sampled
  container memory. RSS is a process high-water mark, including loading;
  container memory also includes filesystem cache and the suite process.
- Transformer and VAE decode tensor shapes, actual output dimensions, nonblank
  image statistics, response model/adapter metadata and SHA-256 of every PNG.
- Synchronized inference time, including prompt encoding and per-request LoRA
  loading/attachment/removal, excluding initial checkpoint loading, downloads
  and PNG encoding. These are direct-runtime times, not serverless API latency.

Here **cold** means the first request in a fresh pipeline process with weights
already downloaded. It does not mean a fresh Pod, empty OS cache, or a timed
checkpoint download. The following request uses the same pipeline process.

These measurements describe one output, a fixed short prompt, square canvases,
the default Base/Turbo CFG/steps and CPU offload on the named GPUs. They do not
establish limits for larger batches, other prompt lengths/aspect ratios,
different CFG/offload settings or other GPU architectures. Reaching the API
ceiling does not find the physical maximum resolution of either GPU.

## Preserved artifacts

- [`measurements.json`](measurements.json), [`measurements.csv`](measurements.csv):
  every observed case and the confirmed maxima.
- [`vram-and-time.png`](vram-and-time.png), [`vram-and-time.pdf`](vram-and-time.pdf):
  observed sweep curves and fresh/warm maximum repeats without extrapolation.
- `results/<hardware>/`: raw event logs, package freeze, GPU report, model/LoRA
  manifests, source hashes, compressed original process logs (`*.log.gz`), selected runtime log lines,
  completion status and Global SHA-256
  verification. Full output PNGs remain on Global volume
  `cmv1ht9mt000007lb00es0w9n`, under
  `/workspace/benchmarks/2026-10-19-moss/<hardware>/`.
- [`harness/benchmark.py`](harness/benchmark.py) and
  [`harness/run_suite.py`](harness/run_suite.py): the measured scripts, adapted
  from the preserved 2026-10-18 harness without changing worker inference.
- [`collect.py`](collect.py): verifies the completed Global files and creates
  metadata and inspection-preview archives before teardown. Optional
  `--preview-side 1024` creates JPEG thumbnails for a smaller transfer; reported
  dimensions and hashes always refer to the unmodified original PNGs on Global.
- [`summarize.py`](summarize.py): validates shapes, adapter revisions, output
  hashes, fresh-process cold/warm confirmation and headroom before reporting.

## Reproduction

Create a fresh dedicated Pod using the exact image above, the intended GPU and
CUDA-compatible driver, enough host RAM and local disk, and the existing Global
volume mounted at `/workspace`. Override the worker start command with
`sleep infinity`. Provide HF credentials through the environment. Use an empty
results directory and a unique Global label for a new experiment.

Copy the two harness files and `protocol.json` into `/opt/moss-bench`, then run:

```sh
nohup python -u /opt/moss-bench/run_suite.py --label l40s-48gb \
  > /opt/moss-bench/suite.log 2>&1 &
```

Observe `/opt/moss-bench/status.json` and `results/events.jsonl`; do not launch a
second run when telemetry is interrupted. The suite fsyncs and hash-checks
Global snapshots throughout execution. After status becomes `complete`, run
`collect.py` inside the Pod and download its two archives before removing the
Pod. Extract metadata into `results/<hardware>` locally, then use a separate
analysis environment with Matplotlib to run `summarize.py`.

This experiment changes neither production inference nor the output-limit
policy. MOSS uses explicit `width`/`height`; shared `reference_megapixels` does
not select its generation canvas.
