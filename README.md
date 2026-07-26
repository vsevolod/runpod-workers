# runpod-workers

Monorepo for custom [RunPod](https://www.runpod.io/) serverless workers.

## Workers

| Path | Description | Status |
|------|-------------|--------|
| [`workers/krea2`](workers/krea2/) | Krea 2 Turbo FP8 image generation (thin Python handler) | MVP |
| [`workers/joycaption`](workers/joycaption/) | JoyCaption Beta One image captioning (thin Python handler) | MVP |
| `workers/shared/` | Shared utilities (reserved) | empty |

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

```bash
export RUNPOD_API_KEY=...
python scripts/krea2_image.py --prompt "a red fox" -o out.png
python scripts/joycaption.py -i photo.jpg -o caption.txt
```

## Tasks / plans

See [`tasks/`](tasks/) for implementation plans (e.g. Krea2 FP8).
