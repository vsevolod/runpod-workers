# MOSS-Image backend

Generation-only MOSS-Image-SFT in the common image worker. Select
`input.model: "moss_image"`. This directory is an inference backend, not a
standalone release image. See [the common worker](../image/README.md).

```json
{
  "input": {
    "model": "moss_image",
    "type": "image_generate",
    "prompt": "A red fox in a snowy forest",
    "width": 1024,
    "height": 1024,
    "seed": 42,
    "model_options": {"variant": "turbo"}
  }
}
```

| Setting | `base` | `turbo` (default) |
|---|---|---|
| Checkpoint | MOSS-Image-SFT | Same SFT checkpoint |
| Steps | 50 | 8 |
| CFG | 3 | Exactly 1 (no guidance) |
| Sampler | Euler | Euler |
| Shift | 5 | 5 |
| Negative-branch skip layers | `[22]` | `[]` |
| DMD adapter | None | `lora_latest.safetensors` at 0.7, unmerged |

`base` means ordinary **SFT** without the accelerator; it does not select
MOSS-Image-Base weights. Switching variants retains the same pipeline and
activates/removes the DMD adapter per request. The built-in adapter is listed
in the Turbo response's `loras` with its effective strength.

`model_options` accepts `variant`, positive finite `shift`, and
`uncond_skip_layers` (unique integers 0..41). Omitted/null skip layers select
the variant default; `[]` disables skipping. Other options are errors.
Dimensions default to 1024x1024, must be multiples of 16 in 256..4096, and
must fit 4,194,304 pixels. These are input limits, not calibrated VRAM bounds.
The shared `reference_megapixels` field does not change generation dimensions.
The [2026-10-19 capacity run](../../docs/benchmarks/2026-10-19-moss/README.md)
confirmed single-image square generation through 2048x2048 in Base/Turbo on
RTX 4090 24 GB and L40S 48 GB with CPU offload. Output-limit behavior still
uses explicit dimensions rather than a shared reference-budget table row.

Common `num_inference_steps` can override the default (1..200); Turbo counts
other than 8 are experimental. Base accepts finite nonnegative CFG; exactly
1 disables the negative branch. Turbo rejects other CFG values, nonblank
negative prompts and nonempty skip layers. JSON layouts can be supplied as
the prompt string without an enhancer or rewrite stage.

`image_edit`, nonempty `images`, and additional user `loras` are rejected
before loading. Only the built-in DMD adapter is currently supported.
Output delivery, errors, seed handling and `num_images` (1..4) follow the
[shared API](../shared/api/README.md).

## Runtime and storage

- Python 3.12; image-specific Torch 2.9.1/cu128 + torchvision 0.24.1,
  Transformers 5.20.1 and the common image's pinned Diffusers commit/patch.
- BF16 DiT and Qwen3.5-9B encoder; FP32 FLUX.2 small VAE, noise and Euler
  updates, following the upstream pipeline.
- `MOSS_MODEL_DIR`: `/models/moss_image_sft` on container disk.
- `MOSS_OFFLOAD`: `cpu` (default, component offload) or `none` (all CUDA).
  This is independent of Qwen's `OFFLOAD` setting.
- `HF_HOME` and `HF_TOKEN` use the shared downloader/cache conventions.
- SFT revision: `2c7954437269461767f533b006823e8f85973a60` in
  `OpenMOSS-Team/MOSS-Image-SFT`.
- DMD revision: `f95df83b5e90ae91ea412434b3f9b7f1c8c6314f` in
  `lodestones/MOSS-Image-SFT-8-steps-dmd`.

The selected backend downloads only JSON/tokenizer/weight data for the five
components. The source pipeline is vendored at the same immutable revision;
downloaded Python files and remote custom code are never executed. Manifests
record repository/revision/file sizes, reuse complete local snapshots, and
repair missing/truncated files. SFT files total approximately 30 GB; Turbo's
additional adapter is approximately 1.19 GB on disk. Both variants share weights.

The DMD adapter downloads on the first Turbo request, under `turbo/` in the
model directory. Its CPU cache is retained until MOSS unloads. Base requests
do not download the accelerator. No data-space sampler, step embedder or
latent-mixing files are loaded. Comfy-Org INT8/ConvRot weights are not used.

For explicit prefetch:

```bash
python workers/moss_image/download_weights.py --output /models/moss_image_sft --turbo
```

## CLI and validation

```bash
python scripts/moss_image.py --prompt 'A red fox' -o moss.png
python scripts/moss_image.py --variant base --prompt 'A poster reading "HELLO"' -o poster.png
python scripts/moss_image.py --request-id JOB_ID -o moss.png
python -m unittest discover -s workers/moss_image/tests -p 'test_*.py'
```

The client uses the common endpoint and supports `ENDPOINT_ID` overrides.
Deploy the candidate image before sending MOSS requests to an existing endpoint.
CPU tests exercise a tiny real MOSS transformer, scheduler, text encoder and
FLUX.2 VAE plus LoRA math/cleanup, local loading and request routing. They do
not establish real-checkpoint image quality, GPU speed, 24 GB fit, or production
model switching. Validate the exact CUDA image and follow the preserved
[calibration procedure](../../docs/benchmarks/image-megapixels.md) before
claiming supported capacities.

## Verified deployment

The [2026-10-19 deployment](../../docs/deployments/2026-10-19-moss-image.md)
completed eight API requests on one RTX 4090 24 GB using the exact published
CUDA image: MOSS Base/Turbo and repeats, Krea Identity V2, Qwen Turbo/Base,
and MOSS Turbo after reloading. MOSS outputs were 1024x1024; warm API times
were 67.8s for Base and 27.3s for Turbo. Repeated MOSS outputs matched their
respective first outputs byte-for-byte, including after family switches.
This deployment check is separate from the subsequent
[40-generation capacity calibration](../../docs/benchmarks/2026-10-19-moss/README.md).

## Sources and licenses

- [SFT model and pipeline](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT).
- [Sampling guide](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md).
- [DMD model](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd).
- [DMD demo's current ordinary-Euler route](https://huggingface.co/spaces/hugging-apps/moss-image-8-step-dmd/blob/main/app.py).

See [NOTICE](NOTICE) and [the upstream license](LICENSES/MOSS-Image-APACHE-2.0.txt).
