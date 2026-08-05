# runpod-workers

Monorepo for custom [RunPod](https://www.runpod.io/) serverless workers.

## Workers

| Path | Description | Status |
|------|-------------|--------|
| [`workers/krea2`](workers/krea2/) | Krea 2 Turbo FP8 image generation (thin Python handler) | MVP |
| [`workers/joycaption`](workers/joycaption/) | JoyCaption Beta One image captioning (thin Python handler) | MVP |
| `workers/minimax_h3_comfy/` | MiniMax H3 T2V via headless ComfyUI (native nodes, pruned int8) | planned — see design/plan |
| [`workers/lora_downloader`](workers/lora_downloader/) | CivitAI LoRA download to network volume (CPU) | MVP |
| `workers/shared/` | Shared utilities (reserved) | empty |

MiniMax H3 design/plan (not implemented yet):

- [`docs/superpowers/specs/2026-08-05-minimax-h3-comfyui-serverless-design.md`](docs/superpowers/specs/2026-08-05-minimax-h3-comfyui-serverless-design.md)
- [`docs/superpowers/plans/2026-08-05-minimax-h3-comfyui-serverless.md`](docs/superpowers/plans/2026-08-05-minimax-h3-comfyui-serverless.md)

## Deploy

Each worker has its own `Dockerfile`. On RunPod **Deploy with GitHub**, set:

- **Dockerfile path** → e.g. `workers/krea2/Dockerfile`
- **Build context** → repository root

Weights should live on a **Network Volume** (not in the image) unless you intentionally bake them.

## Remote test clients

Local CLI scripts (need `RUNPOD_API_KEY`; endpoint IDs are hardcoded, overridable via `ENDPOINT_ID`):

| Script | Endpoint (default) | Usage |
|--------|--------------------|--------|
| [`scripts/krea2_image.py`](scripts/krea2_image.py) | `9zb0wyo61ck3wk` | text2img / edit / fetch → PNG |
| [`scripts/joycaption.py`](scripts/joycaption.py) | `yn0krhztuguxxm` | image → caption text |
| [`scripts/download_lora.py`](scripts/download_lora.py) | set `ENDPOINT_ID` | version ids → volume |

```bash
export RUNPOD_API_KEY=...
python scripts/krea2_image.py --prompt "a red fox" -o out.png
python scripts/joycaption.py -i photo.jpg -o caption.txt
```

## Tasks / plans

See [`tasks/`](tasks/) for implementation plans (e.g. Krea2 FP8).
