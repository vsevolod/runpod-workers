# Image component prefetch deployment — 2026-10-19

Endpoint `9zb0wyo61ck3wk` (`image-workers`) now runs the shared staged
component download/loading lifecycle for MOSS, Qwen Image 2.1 and Krea 2.
The CPU downloader publishes components as they become ready; the worker
starts loading each component while the downloader fetches the following ones.
Selected built-in adapters participate in the same lifecycle. Cold MOSS Turbo
prefetches its DMD LoRA; Base does not download that adapter. The contract is
recorded in [AGENTS.md](../../AGENTS.md).

## Release

- Source: `0a98b047854e549d830234dfeb525d02cfabccd6` on `fix/image-component-prefetch`.
- [CI build and CUDA-image verification](https://github.com/vsevolod/runpod-workers/actions/runs/39310676228): both image jobs succeeded.
- Image: `ghcr.io/vsevolod/runpod-workers-image@sha256:4a971438c674fe6542934c14272701a4272c8c4cead234280349c37642afed5c`.
- The registry's OCI source revision matches the source commit; anonymous pull verified.
- 406 tests passed locally on CPU and inside the published CUDA image:
  shared 41, image 21, Krea 187, Qwen 139, MOSS 18.
- Torch 2.9.1+cu128, torchvision 0.24.1+cu128 and Transformers 5.20.1;
  dependency pins, precision, mmap and CPU offload settings were retained.

## Real GPU checks

All seven requests completed on worker `imphkih9z27zvk`, NVIDIA GeForce
RTX 4090, with the exact image above. Actual CUDA VRAM was 23.5165 GiB.
MOSS used 1024x1024, seed 42, Base 50/CFG 3 and Turbo 8/CFG 1, DMD strength 0.7.
Qwen used 768x1024, seed 42, three repeated portrait references and Lenovo LoRA
at strength 1, Base 40 steps and Turbo 8 steps. Krea used Identity Edit V2,
768x1024, seed 42, two repeated references and eight steps.

All images were decoded, matched the requested dimensions and passed a
nonblank check. The five unique outputs were visually inspected. Every output
matched the corresponding fixture result from the previous deployment
byte-for-byte. MOSS Turbo also matched across cold load, warm use and a return
from another model family. These fixtures do not establish multi-identity quality.

| Case | Steps | API execution seconds | Output SHA-256 |
|---|---:|---:|---|
| moss-turbo-cold | 8 | 188.06 | `2f36899898fc54093e00ba6c02970fc4d694da7949167670b4ff5fc1fff76ce2` |
| moss-base-warm | 50 | 68.02 | `ddc227b4cec55f45fb864cd9051efb66f1f4743372b88ae99d9a1e85cbf88480` |
| moss-turbo-warm | 8 | 26.69 | `2f36899898fc54093e00ba6c02970fc4d694da7949167670b4ff5fc1fff76ce2` |
| qwen-base-cold | 40 | 156.14 | `67e50d8e95498770f419310f7412ba1b90b3171353075e7269df9cf985761e15` |
| qwen-turbo-cold | 8 | 151.99 | `76c3d281e8f5e9d0628a5ddb3c55a2bc4ad67f25c15dc9cba61dd8b81e8e5aaf` |
| krea-identity | 8 | 130.06 | `b5cd2399d7764532d7e1db44e3a5a40af6eda2e187c9c50e4591f5762da648cf` |
| moss-turbo-return | 8 | 30.85 | `2f36899898fc54093e00ba6c02970fc4d694da7949167670b4ff5fc1fff76ce2` |

API execution includes download and initialization where needed. The initial
MOSS request also waited 232.14 seconds in the provider queue; queue time is
excluded from the execution column. These are smoke timings, not a controlled
performance comparison. No additional GPU capacity calibration was run.
The committed MOSS budget table is deployed (the 24 GB row resolves to
4.194304 MP), but these MOSS smoke outputs explicitly requested 1.048576 MP.

## Observed component overlap

Cold MOSS Turbo logs show the worker loading transformer weights at
19:40:36.051–19:40:36.116 UTC while the text encoder was downloading. Encoder
loading at 19:41:36.161–19:41:38.432 overlapped the VAE/scheduler download.
DMD download ran from 19:41:40.845 to 19:41:50.647, overlapping VAE loading
at 19:41:40.869–19:41:40.927. DMD then loaded from the prepared local cache.
Warm MOSS Base/Turbo switching did not download the adapter again.

Qwen Base transformer loading at 19:46:02.179–19:46:02.237 UTC overlapped
text encoder downloading, which finished at 19:46:20.107. The evidence includes
component readiness/loading logs for Krea and both Qwen variants as well.
These loader intervals include CPU construction/mmap; they do not measure
full GPU weight transfer or establish a total cold-start speed improvement.

## Provider configuration

Final endpoint version 52, min=0, max=1, idle timeout 10s, container disk 200 GB.
The temporary 600s idle timeout used for verification was restored. GPU
selector, CUDA restrictions, volume mounts and secrets were preserved.
RunPod reported two throttled initialization attempts alongside the active
worker; the seven completed requests used the active worker on the new digest.

[Sanitized evidence](2026-10-19-image-component-prefetch.json) contains job IDs,
metadata, hashes, selected runtime logs, overlap intervals, baseline comparisons
and final provider readbacks. Other users' interleaved jobs are excluded.
Raw images and private provider snapshots remain outside the repository.
