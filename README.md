# runpod-workers

Monorepo for custom [RunPod](https://www.runpod.io/) serverless workers.

## Workers

| Path | Description | Status |
|------|-------------|--------|
| [`workers/krea2`](workers/krea2/) | Krea 2 Turbo FP8 image generation (thin Python handler) | MVP |
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

Each worker has its own `Dockerfile`. On RunPod **Deploy with GitHub**, set:

- **Dockerfile path** → e.g. `workers/krea2/Dockerfile`
- **Build context** → repository root

Qwen Image and Krea2 download base weights to **container disk** at worker startup
using [`shared/model_download.py`](workers/shared/model_download.py). Their
Dockerfiles copy the shared package into each image, so the build context must
remain the repository root. Other workers keep their existing storage setup.

The shared module owns Hub transfers, local cache configuration, token handling,
download timing and file aliases. Each worker's `download_weights.py` keeps its
model selection, validation and CLI-specific options. Existing CLI commands and
the download-before-handler startup order are preserved.

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
