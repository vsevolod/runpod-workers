# Gliese Qwen3.5 9B caption candidate, 2026-10-21

The caption worker now uses `prithivMLmods/Gliese-Qwen3.5-9B-Abliterated-Caption`
at revision `ead05a2ba36c2e68cc03669414210bde75988185` instead of JoyCaption.
The existing request and response fields, endpoint/client names and directory
are retained. The successful response also includes `model_revision`.
This report records a validated candidate; the production endpoint was not updated.

## Source, tests and image

Runtime source commit: `64e8b1280684a378ee2a10ca3a222d5b5740ce6c`.
[GitHub build and validation](https://github.com/vsevolod/runpod-workers/actions/runs/39494705607)
passed for both workers, including dependency checks, startup packaging, runtime
imports and all 24 caption tests inside the CUDA image. Local CPU suites passed
separately: shared 41, image 21, Krea 187, Qwen 139, MOSS 18 and caption 24.

```text
ghcr.io/vsevolod/runpod-workers-joycaption@sha256:252a9232c88911e4b9bba6bd041750acd89d1cee15dacfc55bf93aaff1c96e9c
```

The staged loader starts the shared CPU-only downloader before heavy imports,
loads a meta model one validated shard at a time directly to GPU and waits for
successful downloader exit before serving. A controlled subprocess test blocks
its second shard until the first has actually loaded. Tests also cover later
failure, a nonzero exit after `all`, cleanup, truncated files, retry reuse and
revision provenance. Tiny-model tensors, logits and generated tokens match the
ordinary Transformers loader. Independent review additionally checked matching
multimodal logits with real image tensors.

## Real GPU check

A temporary RunPod Pod ran the immutable image above. A startup command installed
OpenSSH for test access and preserved the image's CUDA/model environment for SSH
commands; no Python runtime packages or model code were installed or patched.
The harness intercepted SDK registration and called the actual handler, so this
check does not establish production Serverless SDK startup or API behavior.

Execution occurred on **2026-10-20 at approximately 21:02–21:04 UTC**
(21 October in Europe/Moscow).

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb, capability 12.0.
- CUDA-reported total VRAM: 25,367,150,592 bytes.
- Python 3.12.15, Torch 2.8.0+cu128, torchvision 0.23.0+cu128,
  Transformers 5.20.1, Accelerate 1.15.0.
- All 9,409,813,744 model parameters were BF16 on GPU; no quantization or CPU offload.
- The local checkpoint was initially absent. Three downloaded shards totalled
  18,819,721,232 bytes.
- Download/import/load pipeline from `handler.main()` to registration: **55.46 s**.
  This excludes Pod provisioning, image pull, SSH setup and initial handler imports.
- The downloader reported 54.52 s. Processor loading overlapped the first shard
  transfer. Shard 1 loaded in 2.19 s before shard 2 was ready; shard 2 loaded in
  1.61 s before shard 3 was ready. The final downloader-exit barrier completed.

Four requests used the preserved synthetic 1024x1024 image: a red square on a blue
background, with the default 1,048,576-pixel processor cap. All returned captions
identifying both colors. The two greedy results were identical.

| Case | Temperature | Maximum new tokens | Elapsed seconds | Peak Torch allocation, GiB |
|---|---:|---:|---:|---:|
| First request | 0 | 64 | 6.04 | 18.152 |
| Warm request | 0.6 | 256 | 14.58 | 18.152 |
| Warm request | 0.6 | 512 | 14.94 | 18.152 |
| Repeated greedy request | 0 | 64 | 4.72 | 18.152 |

The 512-token limit is a cap; that answer finished earlier. Peak Torch reserved
memory was 19,551,748,096 bytes. Process maximum host RSS was 12,884,037,632 bytes;
this is not a measurement of total container RAM or whole-device memory.

These observations cover this GPU, fixture, prompt and settings. They do not
establish capacity or caption quality for arbitrary images. No GPU calibration
or comparison against sequential downloading was performed, so the amount of
cold-start improvement remains unmeasured. Missing `causal_conv1d` and
`flash-linear-attention` used the runtime's correct reference PyTorch kernels;
optimized-kernel performance was not tested.

## Evidence and cleanup

The [executed harness](2026-10-21-gliese-caption/gpu_smoke.py),
[log](2026-10-21-gliese-caption/gpu_smoke.log),
[results](2026-10-21-gliese-caption/results.json) and
[metadata/evidence](2026-10-21-gliese-caption/evidence.json) are preserved.
Artifacts were SHA-256 verified against the Pod copies, and process exit was 0.

Test Pod `h6xjzzp4vc3vuv` was terminated after saving results; a subsequent account
Pod listing was empty. The production caption endpoint remains unchanged.

The cancelled local build was replaced by the GitHub build. Its build cache,
the temporary CPU environment and downloaded metadata were removed; the uv cache
was also cleared. Local free disk increased from approximately 5 to 20 GiB.
Running containers, volumes and workspace files were preserved.
