# MOSS Image deployment — 2026-10-19

Endpoint `9zb0wyo61ck3wk` (`image-workers`) now includes `model: "moss_image"`.
MOSS `variant: "base"` is ordinary SFT; `"turbo"` uses the same SFT checkpoint
with the pinned eight-step DMD LoRA at strength 0.7, applied unmerged.

## Release

- Source: `d95b30908773c9d0d5d49ad69ede571ad1caffe5` on `feat/moss-image`.
- [CI build and CUDA-image verification](https://github.com/vsevolod/runpod-workers/actions/runs/39272476342).
- Image: `ghcr.io/vsevolod/runpod-workers-image@sha256:fbe62a59c6fbc9099c1f48fa49529eddb34446ecd7b52cb0830f1850ab35b7a9`.
- Torch 2.9.1+cu128 / torchvision 0.24.1+cu128; Transformers 5.20.1;
  pinned common Diffusers commit and native-reference patch.
- 392 tests passed locally on CPU and all 392 passed inside the published CUDA
  image in CI. Runtime/dependency checks passed.
- Model/adapter revisions are included in every MOSS job's metadata and
  pinned in [the backend](../../workers/moss_image/README.md).

## Real GPU checks

All eight requests completed on worker `ysi6f1pyi48b9x` (`NVIDIA GeForce RTX 4090`)
with the exact image above. MOSS used 1024x1024, seed 42, Base 50/CFG 3,
Turbo 8/CFG 1. Krea used Identity Edit V2 with two references; Qwen used
Lenovo with three references in Turbo and Base, at 768x1024. All images were
decoded, matched requested dimensions and passed a nonblank check. The five
unique outputs were visually inspected; repeated MOSS results matched their
respective original images byte-for-byte. Reference cases repeat one portrait
fixture and do not establish multi-identity quality.

| Case | Steps | API execution seconds | Output SHA-256 |
|---|---:|---:|---|
| moss-base-cold | 50 | 215.42 | `ddc227b4cec55f45fb864cd9051efb66f1f4743372b88ae99d9a1e85cbf88480` |
| moss-turbo-first | 8 | 35.11 | `2f36899898fc54093e00ba6c02970fc4d694da7949167670b4ff5fc1fff76ce2` |
| moss-turbo-warm | 8 | 27.31 | `2f36899898fc54093e00ba6c02970fc4d694da7949167670b4ff5fc1fff76ce2` |
| moss-base-after-turbo | 50 | 67.80 | `ddc227b4cec55f45fb864cd9051efb66f1f4743372b88ae99d9a1e85cbf88480` |
| krea-identity | 8 | 143.48 | `b5cd2399d7764532d7e1db44e3a5a40af6eda2e187c9c50e4591f5762da648cf` |
| qwen-turbo | 8 | 100.74 | `76c3d281e8f5e9d0628a5ddb3c55a2bc4ad67f25c15dc9cba61dd8b81e8e5aaf` |
| qwen-base | 40 | 109.82 | `67e50d8e95498770f419310f7412ba1b90b3171353075e7269df9cf985761e15` |
| moss-turbo-return | 8 | 40.24 | `2f36899898fc54093e00ba6c02970fc4d694da7949167670b4ff5fc1fff76ce2` |

API execution includes checkpoint/adapter download and initialization where
needed. These are smoke timings, not a controlled performance benchmark.
Comparisons of the same seed before/after Turbo and model switches, plus
CUDA allocations after unloading each family, are in the evidence JSON.
No capacity beyond the tested requests is claimed.

## Provider configuration

Endpoint version 51; min=0, max=1,
idle timeout 10s, container disk 200 GB.
The temporary 600s idle timeout used to retain one worker during verification
was restored. L4 was temporarily excluded after provider GPU throttling; the
original GPU selector was restored after verification. CUDA restrictions,
volume mounts, model paths and existing secrets were preserved. MOSS uses component CPU offload.

[Sanitized evidence](2026-10-19-moss-image.json) includes job IDs, GPU/image
readbacks, metadata, hashes, selected runtime logs and image comparisons.
Raw images and private provider snapshots remain outside the repository.
