# runpod-workers

Custom RunPod serverless workers.

## Workers

| Path | Description |
|---|---|
| [`workers/image`](workers/image/) | One request-selected image worker: Krea 2 and Qwen Image 2.1 Base/Turbo |
| [`workers/krea2`](workers/krea2/) | Krea inference backend, quantization and V2 edit adapters |
| [`workers/qwen_image_21`](workers/qwen_image_21/) | Qwen inference backend, Base/Turbo and additional LoRAs |
| [`workers/joycaption`](workers/joycaption/) | JoyCaption Beta One captioning |
| [`workers/minimax_h3_comfy`](workers/minimax_h3_comfy/) | MiniMax H3 headless ComfyUI worker |
| [`workers/lora_downloader`](workers/lora_downloader/) | CPU downloads of CivitAI adapters to the shared volume |
| [`workers/shared`](workers/shared/) | Common API, downloads, adapter cache and megapixel policy |

MiniMax H3 design/plan:

- [`docs/superpowers/specs/2026-08-05-minimax-h3-comfyui-serverless-design.md`](docs/superpowers/specs/2026-08-05-minimax-h3-comfyui-serverless-design.md)
- [`docs/superpowers/plans/2026-08-05-minimax-h3-comfyui-serverless.md`](docs/superpowers/plans/2026-08-05-minimax-h3-comfyui-serverless.md)
- Worker: [`workers/minimax_h3_comfy/`](workers/minimax_h3_comfy/) (`PINS.md`, frozen API workflow, handler)

## Deploy

Krea and Qwen share endpoint **`9zb0wyo61ck3wk`**, named `image-workers`.
Select `input.model`: `krea2` or `qwen_image_21`. Omitting it selects Krea.
Qwen Base/Turbo remains `model_options.variant`.
The [shared API](workers/shared/api/README.md) documents common fields,
model options and responses; the [image worker](workers/image/README.md)
documents packaging and configuration.

[Build Runpod workers](.github/workflows/build-workers.yml) publishes:

- `ghcr.io/vsevolod/runpod-workers-image:<commit-sha>`
- `ghcr.io/vsevolod/runpod-workers-joycaption:<commit-sha>`

Changes to the image worker, either backend, shared code, bundled image clients
or workflow on `master` trigger the build. A manual build uses:

```bash
gh workflow run build-workers.yml --ref master -f publish=true
gh run list --workflow build-workers.yml
```

Build locally from the repository root:

```bash
docker build -f workers/image/Dockerfile -t runpod-image .
```

After CI passes, set the RunPod endpoint to the exact published image digest.
GitHub publication does not update the endpoint automatically. Keep its Docker
start command empty to use `CMD`, or use `sh /app/workers/image/start.sh`.
The image imports the handler without choosing a checkpoint; the request starts
only its selected model's downloads and loading. Switching models closes the
old backend first. Downloaded checkpoints stay on container disk for reuse.

Use a 200 GB container disk for Krea and both Qwen variants. Attach the existing
Global Volume at `/runpod-volume` for additional LoRAs. Model weights and the HF
cache live under `/models`; selected adapters are staged in a local cache.
The model/VRAM megapixel policy and its calibration procedure are documented in
[Image megapixel benchmarks](docs/benchmarks/image-megapixels.md).

Other workers retain their own Dockerfiles and deployment settings. For
RunPod GitHub builds, use the repository root as Docker build context.

## Remote test clients

All clients need `RUNPOD_API_KEY`.

| Script | Default endpoint | Purpose |
|---|---|---|
| [`scripts/krea2_image.py`](scripts/krea2_image.py) | `9zb0wyo61ck3wk` | Krea generation/edit; sends `model=krea2` |
| [`scripts/qwen_image_21.py`](scripts/qwen_image_21.py) | `9zb0wyo61ck3wk` | Qwen generation/edit; sends `model=qwen_image_21` |
| [`scripts/joycaption.py`](scripts/joycaption.py) | `yn0krhztuguxxm` | Caption images |
| [`scripts/download_lora.py`](scripts/download_lora.py) | Set `ENDPOINT_ID` | Download adapters to the volume |

`ENDPOINT_ID` overrides the default for either image client.

```bash
python scripts/krea2_image.py --prompt "a red fox" --lora '' -o krea.png
python scripts/qwen_image_21.py --prompt "a red fox" --variant turbo -o qwen.png
```

## Tasks / plans

See [`tasks/`](tasks/) and [`docs/superpowers/plans/`](docs/superpowers/plans/).
