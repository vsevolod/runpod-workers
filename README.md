# runpod-workers

Monorepo for custom [RunPod](https://www.runpod.io/) serverless workers.

## Workers

| Path | Description | Status |
|------|-------------|--------|
| [`workers/krea2`](workers/krea2/) | Krea 2 Turbo FP8/INT8 generation + Identity/General Edit V2 (thin Python handler) | MVP |
| [`workers/qwen_image_21`](workers/qwen_image_21/) | Qwen-Image-2.1-Turbo text-to-image and multi-ref edit (Diffusers, official 8-step checkpoint) | MVP |
| [`workers/joycaption`](workers/joycaption/) | JoyCaption Beta One image captioning (thin Python handler) | MVP |
| [`workers/minimax_h3_comfy/`](workers/minimax_h3_comfy/) | MiniMax H3 T2V via headless ComfyUI (native nodes, pruned int8) | scaffold — headless GPU smoke pending |
| [`workers/lora_downloader`](workers/lora_downloader/) | CivitAI LoRA download to network volume (CPU) | MVP |
| [`workers/shared/`](workers/shared/) | Shared model downloads and local HF cache setup for Qwen Image and Krea2 | used by both workers |

MiniMax H3 design/plan:

- [`docs/superpowers/specs/2026-08-05-minimax-h3-comfyui-serverless-design.md`](docs/superpowers/specs/2026-08-05-minimax-h3-comfyui-serverless-design.md)
- [`docs/superpowers/plans/2026-08-05-minimax-h3-comfyui-serverless.md`](docs/superpowers/plans/2026-08-05-minimax-h3-comfyui-serverless.md)
- Worker: [`workers/minimax_h3_comfy/`](workers/minimax_h3_comfy/) (`PINS.md`, frozen API workflow, handler)

## Deploy

The [shared image API](workers/shared/api/README.md) is the complete request and
response contract for Krea 2 and Qwen Image 2.1. Common fields use the same names;
model-specific controls live under `model_options`, with existing flat keys
preserved as aliases.

The shared `reference_megapixels: "max"` policy and its GPU calibration procedure
are documented in [Image megapixel benchmarks](docs/benchmarks/image-megapixels.md),
including raw measurements and the scripts used for the L40S 48 GB run.

### Qwen Image and Krea2: build on GitHub, run on Runpod

[Build Runpod workers](.github/workflows/build-workers.yml) builds both complete
CUDA images on standard GitHub-hosted Ubuntu runners. A push to `master` that
changes either worker, `workers/shared`, or the workflow builds and publishes
both images to GitHub Container Registry. You can also run it manually:

```bash
gh workflow run build-workers.yml --ref master -f publish=true
gh run list --workflow build-workers.yml
```

The image tag is the full commit SHA:

```text
ghcr.io/vsevolod/runpod-workers-qwen_image_21:<commit-sha>
ghcr.io/vsevolod/runpod-workers-krea2:<commit-sha>
```

The workflow checks installed dependencies, downloader/bootstrap packaging,
shared coordination tests and model runtime imports before publication. These
checks do not load weights or prove GPU inference. Base weights remain outside
the images and download at worker startup.

Deploy a successfully published image through Runpod's **Import from Docker
Registry** flow, or update an existing endpoint's template to the exact image
tag/digest. Keep existing endpoint IDs when updating templates. Ensure the image
is publicly pullable or configure a read-only registry credential on Runpod.
Leave the Docker start command empty to use the image's `CMD`.

Use a **120 GB container disk** for both Qwen variants or **80 GB** for Krea,
`MODEL_DIR=/models/qwen_image_21_turbo` for Qwen or `/models/krea2` for Krea,
and `HF_HOME=/models/hf`. Preserve the attached LoRA
volume, `LORA_DIR`, GPU selection, quantization and offload settings.

During a migration, set `workersMin=0` and `workersMax=0` to stop GPU allocation
and retain the previous template/settings for rollback. Restore the saved
worker limits once the intended image is configured, then verify a real GPU
request and startup logs. A published image is not automatically deployed by
this workflow; update the endpoint explicitly for each release.

### Other workers: Runpod GitHub integration

Each worker has its own `Dockerfile`. On Runpod **Deploy with GitHub**, set:

- **Dockerfile path** → e.g. `workers/krea2/Dockerfile`
- **Build context** → repository root

Qwen Image and Krea2 download base weights to **container disk** at worker startup
using [`shared/model_download.py`](workers/shared/model_download.py). Their
Dockerfiles copy the shared package into each image, so the build context must
remain the repository root. Other workers keep their existing storage setup.

The shared module owns Hub transfers, local cache configuration, token handling,
download timing and file aliases. Each worker's `download_weights.py` keeps its
model selection, validation and CLI-specific options. Existing CLI commands and
the local model layout are preserved. A CPU-only downloader process overlaps
downloads with ML imports. Krea2 waits separately for DiT, VAE and text encoder,
so each completed component can load while the next one downloads. Qwen waits
for its validated snapshot after imports, preserving mmap and CPU offload.

The shared `bootstrap.py` owns the download process and cleans it up on loading
failure or shutdown. Readiness signals live in a fresh temporary directory for
each startup. The worker accepts jobs only after every required component is
ready and the downloader has exited successfully.

### Existing endpoint startup commands

For Qwen Image and Krea2, leave the endpoint's Docker start command empty to use
the image's default `CMD`, or set it to `sh /app/start.sh`. An override such as
`python /app/handler.py` or `download_weights.py && handler.py` bypasses the shared
bootstrap and its overlapping downloads.

Only use `sh /app/start.sh` once the endpoint's **active image** contains that
file. A successful push or an endpoint configuration release does not prove the
new image was built and activated. In particular, Qwen's older `57261f6` image
has no `start.sh`. While that image is still active, this transitional Docker
start command preserves its sequential startup and automatically uses the
bootstrap after a new image is activated:

```text
sh -c "if [ -f /app/start.sh ]; then exec sh /app/start.sh; else python -u /app/download_weights.py && exec env LOCAL_FILES_ONLY=1 python -u /app/handler.py; fi"
```

After the new image is active, remove the transitional override so future
deployments follow the image's `CMD`. Check both the repository configuration's
start command and the active endpoint template's `dockerArgs`; updating the
Dockerfile alone does not remove an existing template override.

## Remote test clients

Local CLI scripts (need `RUNPOD_API_KEY`; endpoint IDs are hardcoded unless the table says to set `ENDPOINT_ID`):

| Script | Endpoint (default) | Usage |
|--------|--------------------|--------|
| [`scripts/krea2_image.py`](scripts/krea2_image.py) | `9zb0wyo61ck3wk` | text2img / edit / fetch → PNG |
| [`scripts/qwen_image_21.py`](scripts/qwen_image_21.py) | set `ENDPOINT_ID` | text2img / multi-ref edit / fetch → PNG |
| [`scripts/joycaption.py`](scripts/joycaption.py) | `yn0krhztuguxxm` | image → caption text |
| [`scripts/download_lora.py`](scripts/download_lora.py) | set `ENDPOINT_ID` | version ids → volume |

```bash
export RUNPOD_API_KEY=...
python scripts/krea2_image.py --prompt "a red fox" -o out.png
python scripts/joycaption.py -i photo.jpg -o caption.txt
```

## Tasks / plans

See [`tasks/`](tasks/) for implementation plans (e.g. Krea2 FP8).
