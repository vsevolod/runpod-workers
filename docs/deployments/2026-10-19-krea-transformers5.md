# Krea migration to the Qwen image dependency stack, 2026-10-19

Krea now uses Transformers `5.20.1` and the same pinned Diffusers commit as
Qwen Image 2.1: `da1d3829cf08d4f329b526d89e17cc035c049d8d`.
The existing Krea inference implementation required no changes.

Source commit: `2e1a8736f47307f10fc5f66a31225356f77ceda8`.
[Build and checks](https://github.com/vsevolod/runpod-workers/actions/runs/39238702825) completed successfully.

## Validation

The published CUDA image passed dependency resolution, runtime imports,
187 Krea tests and 41 shared tests with Torch `2.8.0+cu128`, torchvision
`0.23.0+cu128` and CUDA 12.8.

A separate local CPU probe exercised the unchanged Krea conditioner with the
real production tokenizer/processor and tiny random Qwen3-VL weights: text,
one reference and two references. That probe used Torch `2.14.1+cpu` and
checks the encoder interface; production GPU generation below validates the
full checkpoint on Torch 2.8.

## Production deployment

Endpoint: `9zb0wyo61ck3wk`.
Image: `ghcr.io/vsevolod/runpod-workers-krea2@sha256:26a6413dadd77ef93fab60d83485dbe8e86c6e7359559bd7730f237e21eeefea`.
The OCI source revision and the digest were checked against the successful build.
RunPod worker metadata and system logs confirmed this exact image.

Endpoint version is 47. Idle timeout was temporarily 180 seconds for
validation and restored to 10 seconds. Worker limits remain
`min=0`, `max=1`. A full configuration comparison confirmed that only the
image and provider-generated release/template IDs differ from the original.
The Qwen endpoint configuration matches its original snapshot.

## Production API checks

All cases used the current INT8 ConvRot production profile on NVIDIA RTX A5000,
eight steps, seed 42 and a 1 MP resolved reference budget. Every response
contained one decoded, nonblank 768x1024 image; SHA-256 hashes and job/worker
IDs are retained in the evidence. Outputs were visually inspected.

| Case | References | LoRA at strength 1 | Execution | Result |
|---|---:|---|---:|---|
| text | 0 | none | 49.779 s | COMPLETED |
| identity | 2 | krea2_identity_edit_v2 | 98.747 s | COMPLETED |
| general | 1 | krea2_general_edit_v2 | 73.946 s | COMPLETED |

[Machine-readable evidence](2026-10-19-krea-transformers5.json) includes provider readback,
CI checks, response metadata, image hashes and observed runtime stages.
Credentials and reference/output image bytes are excluded.
