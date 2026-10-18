# Image worker deployment, 2026-10-19 Moscow

After explicit deployment authorization, both production endpoints were updated
to the exact immutable images tested in the
[Torch/CUDA validation](../benchmarks/torch28-cu128-upgrade.md).
The deployment occurred on October 18 UTC, October 19 in Moscow.

| Worker | Endpoint | Endpoint version | Image digest |
|---|---|---:|---|
| Krea | `9zb0wyo61ck3wk` | 46 | `sha256:96d5b11d4595d4305e26dd1ff3b42bacb521668987fa8b160ffee57490b998ff` |
| Qwen Image 2.1 | `b1zzsd2eqqdrvp` | 30 | `sha256:2f7b4b7eb5c25ef95a8b017c68ecc84737e9affad510d23c0669bd257e01573e` |

Both endpoints now require minimum CUDA `12.8` and allow host versions
`12.8,12.9,13.0,13.1,13.2,13.3,13.4`. Images contain Torch `2.8.0+cu128`,
torchvision `0.23.0+cu128` and the CUDA 12.8.1 runtime. Qwen's production
startup log also confirmed `torch=2.8.0+cu128 cuda_build=12.8`.

The existing Ampere/Ada GPU selectors, volume mounts, environment settings,
worker limits (`min=0`, `max=1`) and other configuration were preserved.
Before each update there were no queued or executing jobs. Configuration
readback matched the intended image/CUDA changes; all workers returned by the
final provider listing used the new image and had `isStale=false`.

## Production API smoke checks

| Worker | GPU | References | LoRA @ 1.0 | Execution time | Result |
|---|---|---:|---|---:|---|
| Krea INT8 ConvRot | NVIDIA L4 24 GB | 2 | Identity Edit V2 | 218.062 s | COMPLETED |
| Qwen Turbo | RTX A5000 24 GB | 3 | Lenovo | 89.981 s | COMPLETED |

Both edits used eight steps and seed 42 and returned exactly one nonblank
864x1152 image. Image bytes were decoded, dimensions and output metadata checked,
and results visually inspected. Krea omitted `reference_megapixels` to test its
new default; Qwen explicitly supplied `"max"`. Both responses reported
`reference_megapixels="max"`, `resolved_reference_megapixels=1` and actual
CUDA-reported device memory. These checks do not extend the earlier 4 MP
calibration or test every profile again; the prior image validation covered
Base/Turbo and INT8/FP8 separately.

[Provider configuration and smoke evidence](2026-10-19-image-workers.json)
contains job IDs, worker IDs, response metadata and output SHA-256 hashes.
Credentials and reference/output image bytes are excluded.
