# Torch 2.8 / CUDA 12.8 worker image validation

The old image Dockerfiles selected CUDA 12.4 and installed Torch 2.6.0+cu124,
while the earlier L40S megapixel calibration used Torch 2.8.0+cu128. Commit
`938a3238e186a6b8038f3fddaffc94b3ca4c3f76` aligns the image Torch/CUDA pair with
that calibration and prevents later dependency installation from changing it.

Both Dockerfiles now pin the CUDA 12.8.1 Ubuntu 22.04 image digest and install
`torch==2.8.0+cu128` / `torchvision==0.23.0+cu128` from the CUDA 12.8 index.
The shared [constraints](../../workers/shared/torch-cu128.constraints.txt) also
apply to each worker's requirements installation. CI asserts all three installed
versions after dependency installation and checks runtime imports and tests.

[Build and publication CI](https://github.com/vsevolod/runpod-workers/actions/runs/39162351332)
completed successfully for both images. The GPU tests below used these actual
published images, overriding only their start command to prepare a standalone
test Pod; they did not install an alternative ML environment or replace app code.

```text
ghcr.io/vsevolod/runpod-workers-krea2@sha256:96d5b11d4595d4305e26dd1ff3b42bacb521668987fa8b160ffee57490b998ff
ghcr.io/vsevolod/runpod-workers-qwen_image_21@sha256:2f7b4b7eb5c25ef95a8b017c68ecc84737e9affad510d23c0669bd257e01573e
```

## GPU checks

**16/16 actual GPU generations succeeded.** Each table cell below represents
two consecutive requests in one resident pipeline: the first request after
loading and a warm repeat. Each profile starts in a fresh process.

| Profile | Ampere GPU | Ada GPU | Fixed LoRA @ 1.0 | Refs | Steps | Guidance |
|---|---|---|---|---:|---:|---:|
| Krea INT8 ConvRot | RTX 3090 24 GB | RTX 4090 24 GB | Identity Edit V2 | 2 | 8 | 0 |
| Krea FP8 | RTX 3090 24 GB | RTX 4090 24 GB | Identity Edit V2 | 2 | 8 | 0 |
| Qwen Turbo | RTX A5000 24 GB | RTX 4090 24 GB | Lenovo | 3 | 8 | 1 |
| Qwen Base | RTX A5000 24 GB | RTX 4090 24 GB | Lenovo | 3 | 40 | 1 |

All requests used top-level `reference_megapixels: "max"`, resolving to **1 MP**
on these 24 GB cards, automatic 992x992 square output, seed 42 and one output
image. Qwen used CPU offload and the texture-fix VAE. Fixture hashes match across
all four Pods. Real shared parsing, preprocessing, runtime inference, LoRA
attachment, image dimensions and nonblank outputs were checked. Warm outputs
from every profile/GPU combination were also visually inspected.

These are compatibility smoke tests, not a calibration above 1 MP and not a
version-to-version performance comparison. The Community RTX 3090 host had a
150 W power limit and was observed at 210 MHz SM clock under full load, making
its timings unrepresentative of unrestricted RTX 3090 hardware. The other
Torch/CUDA dependencies and hardware details are preserved per Pod below.

The previous 4 MP policy entry remains based on the separate L40S 48 GB run.
These image smoke tests did not rerun that sweep or validate arbitrary extra
LoRAs, reference counts, aspect ratios or CFG settings.

## Evidence and storage

- [Image and hardware manifest](2026-10-18-torch28-cu128/manifest.json).
- [All 16 measurements](2026-10-18-torch28-cu128/measurements.csv).
- Raw events, installed package freezes, GPU reports, fixture hashes and
  completion status: [Ada Krea](2026-10-18-torch28-cu128/ada-krea/),
  [Ada Qwen](2026-10-18-torch28-cu128/ada-qwen/),
  [Ampere Krea](2026-10-18-torch28-cu128/ampere-krea/),
  [Ampere Qwen](2026-10-18-torch28-cu128/ampere-qwen/).
- [Global image verification](2026-10-18-torch28-cu128/global-verification.json).
- Executed [inference harness](2026-10-18-torch28-cu128/harness/benchmark.py)
  and [runner](2026-10-18-torch28-cu128/harness/run_smoke.py), identical across
  all four test environments.
- [Verified Pod shutdown](2026-10-18-torch28-cu128/cleanup.json).

Repository copies of the GPU text reports have normalized whitespace; the
original reports are preserved with the complete results on Global storage.

Complete results, PNG outputs and the executed harness scripts are persisted
at `/workspace/benchmarks/2026-10-18-torch28-cu128/<label>/` on Global volume
`cmv1ht9mt000007lb00es0w9n`. A Community RTX 3090 machine could not mount Global
storage; that failed-start Pod was stopped and recreated without a mount.
Its completed results were transferred with SHA-256 verification, then written
to Global through the A5000 Pod with flush/fsync and hash checks. No model
weights were added to Global storage.

The run started on October 18 UTC and finished on October 19 in Moscow;
directory names use the UTC start date. All test Pods were stopped after their
results were preserved. Production endpoints were not redeployed or edited.
Deployment of these images requires selecting CUDA 12.8 and newer in Runpod's
allowed CUDA versions; current Ampere/Ada GPU model pools remain compatible.

Both production endpoints were subsequently deployed after explicit user
authorization: [deployment and API smoke evidence](../deployments/2026-10-19-image-workers.md).

For a new model, use the [calibration procedure](image-megapixels.md#procedure-for-a-new-image-model-or-gpu)
and run the actual candidate image, recording its digest and runtime versions
before the first measured request.
