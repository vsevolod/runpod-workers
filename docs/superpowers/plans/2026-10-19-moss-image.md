# MOSS Image implementation plan

**Goal:** Add generation-only MOSS-Image-SFT with ordinary and eight-step Turbo modes to the common image worker.

**Architecture:** Register `moss_image` alongside Krea/Qwen; keep one SFT pipeline resident and scope unmerged DMD LoRA activation to each Turbo request. Vendor the official Python pipeline at a pinned revision and load only pinned data files locally. Use the existing ownership manager and image-delivery contract.

**Execution:** Inline in `feat/moss-image`. The accepted design is the preceding conversation; no deployment or capacity claim is included.

## Constraints

- `model_options.variant`: `base` (ordinary SFT, 50 steps, CFG 3) or `turbo` (8 steps, CFG 1), default Turbo.
- Euler, shift 5, dimensions divisible by 16. Ordinary mode skips negative-branch layer 22; Turbo disables guidance/skipping.
- Pinned SFT revision `2c7954437269461767f533b006823e8f85973a60` and DMD revision `f95df83b5e90ae91ea412434b3f9b7f1c8c6314f`; `lora_latest.safetensors` strength 0.7.
- BF16 transformer/encoder, FP32 VAE/noise/latent updates; component CPU offload. No ComfyUI, INT8, image editing, data-space sampler, or new external deployment.
- Image-only Torch 2.9.1/cu128 + torchvision 0.24.1; caption worker retains its existing stack.

## Tasks

- [x] Contract: add MOSS profile/normalizer and routing tests. Reject editing and incompatible Turbo guidance before loading weights; retain explicit seed/size/step settings and JSON prompts.
- [x] Downloads: write tests for pinned file selection, complete snapshots, failed/truncated download recovery, and lazy Turbo adapter. Implement a small manifest-backed store and downloader using the shared download helpers.
- [x] Runtime: test real tiny MOSS transformer generation/decoding, LoRA math, validation, exception cleanup, and Turbo/base switching. Vendor source/license, implement strict unmerged LoRA application and component loading/offload/close.
- [x] Integration: register backend, include it in the common image/CI, add a generation/fetch CLI and payload tests, and document runtime/settings/limitations.
- [x] Verification: run MOSS/shared/common/Krea/Qwen suites, dependency/import checks and a candidate image build when available. Review the complete diff; report local execution separately from unperformed real-model GPU tests.

## Verification evidence

- Final local suites: shared 41, common image 18, Krea 187, Qwen 135, MOSS 11;
  **392 tests passed**, no failures.

- CPU environment: Python 3.12.3, Torch 2.9.1+cpu, torchvision 0.24.1+cpu,
  Transformers 5.20.1, Diffusers da1d3829cf08d4f329b526d89e17cc035c049d8d with
  the repository native-reference patch, Accelerate 1.15.0, safetensors 0.8.0.
- MOSS tests execute real tiny MOSS DiT, Qwen3.5 (linear/full attention),
  FLUX.2 VAE and Accelerate model-offload hooks on CPU, plus adapter math and
  request-scoped cleanup. They do not load released model weights.
- Read-only upstream header check: all 252 rank-128 DMD adapter targets/shapes
  match the pinned full-size architecture instantiated on the meta device.
- CUDA 12.8 Torch 2.9.1 / torchvision 0.24.1 wheel resolution: successful dry run.
- Handler/all backend imports without weights: passed. Downloader/client help,
  dependency compatibility, actionlint, Python syntax and diff whitespace checks: passed.
- Full CUDA Docker build and real-model RunPod generation were not executed.
  The local filesystem had about 15 GB free after test setup; no existing images,
  volumes or unrelated files were removed to make room. CI is configured to build
  and verify the actual image when the branch is published/integrated.
- At implementation handoff no deployment, calibration, commit or push had been performed.
  The user subsequently authorized publication, deployment and real-generation checks.
