# MiniMax H3 — ComfyUI Serverless Worker (Design)

**Дата:** 2026-08-05 (rev 2 after plan review)  
**Статус:** draft for approval (not implementation-approved until Task/Phase 0 smoke)  
**Официальный tutorial:** [MiniMax H3 in ComfyUI](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)  
**Официальный T2V template:** [video_minimax_h3_t2v.json](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json)

## Решение

Thin ModularPipeline / Level-1 inject **отменены**. Product path = **headless ComfyUI** + **native** MiniMax H3 nodes (ComfyUI ≥ 0.30.0), weights Comfy-Org, serverless RunPod.

## Goal

1. Serverless job: product input → native T2V workflow → **MP4 with native stereo audio**.  
2. Default DiT: `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (~19.5 GB) + three companions (TE + 2×VAE).  
3. Output path taken **only** from **SaveVideo** node history metadata (not “largest mp4”).  
4. Delivery: all four `BUCKET_*` → URL; none → limited inline base64; **partial** `BUCKET_*` → **startup error**.

## Non-goals (v1)

- Diffusers ModularPipeline / thin inject.  
- Mandatory **custom node packages** for core T2V (native only).  
- KJNodes / SageAttention (optional later; not in v1 path).  
- Ref2VA / R2V / I2V product modes (T2V first).  
- Raw `input.comfy_workflow` escape hatch.  
- Multi-workflow product surface.

## Architecture (minimal)

```text
start.sh
  → symlink volume models → ComfyUI/models
  → ComfyUI (pinned commit ≥ H3 support) on 127.0.0.1:8188
  → python -u handler.py

handler.py
  → validate product input
  → workflow.py: load pinned API JSON + inject fields
  → POST /prompt → poll /history/{id}
  → resolve SaveVideo output path from history
  → delivery: S3 URL or base64
```

**No** separate `runtime.py` / `comfy_client.py` / multi-module request stack unless forced by size. Prefer:

| File | Role |
|------|------|
| `handler.py` | RunPod entry, Comfy HTTP via `requests`, delivery |
| `workflow.py` | load template, inject, duration/canvas helpers used by inject |
| `download_weights.py` | four exact HF files |
| `workflows/t2va_api.json` | **API-format** export (not UI subgraph template as-is) |
| `start.sh` | process layout |
| `Dockerfile` | pin ComfyUI commit; optional worker-comfyui only as boot pattern |
| `tests/` | 2–3 unit files (inject + delivery + input normalize) |

## Native vs custom nodes

| Layer | v1 |
|-------|-----|
| Core graph | **Native** ComfyUI nodes (`UNETLoader`, `CLIPLoader`, `MiniMaxH3ImageToVideo`, `CreateVideo`, `SaveVideo`, …) after ComfyUI **≥ 0.30.0** / pin with PR #15224 lineage |
| Custom nodes | **Not required** for default path |
| SageAttention | Optional later: package + **KJNodes** `Patch Sage Attention KJ` — out of v1 |

Sources: [Comfy docs MiniMax H3](https://docs.comfy.org/tutorials/video/minimax/minimax-h3) (“ComfyUI natively supports MiniMax H3”; Sage section explicitly optional + KJNodes).

## Exact four weights (T2V)

From official storage layout:

```text
ComfyUI/models/
  diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
  text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
  vae/minimax_h3_video_vae_fp16.safetensors
  vae/minimax_h3_audio_vae_fp32.safetensors
```

HF: [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3).  
Volume bootstrap downloads **only these four** for T2V v1.

## ComfyUI pin (critical)

- Official docs: update ComfyUI to **0.30.0 or later** for H3.  
- [runpod-workers/worker-comfyui](https://github.com/runpod-workers/worker-comfyui) release tags (e.g. 5.8.x) may predate H3 — **do not** assume `…-base` image has new enough Comfy.  
- **Must** pin an explicit **ComfyUI git commit / release** known to include MiniMax H3 native nodes, then either:  
  - build image that clones that pin, or  
  - start from worker-comfyui base **and overwrite/upgrade** ComfyUI to the pin.  
- Document pin in `PINS.md` after Phase 0 verifies headless smoke.

## Workflow artifact

1. Start from official template `video_minimax_h3_t2v` (UI/subgraph form).  
2. On a host with pinned Comfy: **Workflow → Export (API)** → freeze `workflows/t2va_api.json`.  
3. Record **inject map**: node id → input field for `prompt`, `width`, `height`, `duration` (or snapped `length`), `seed` / `noise_seed`.  
4. Record **SaveVideo node id** and exact **history outputs** key path to the written file (filename + subfolder + type).  
5. Phase 0 fails closed if history shape is unknown.

**Forbidden:** pick largest `.mp4` under `output/`.

## Canvas & duration

| Constraint | Source |
|------------|--------|
| Multiple of **32** | official |
| Native canvas: **768 short edge**, cap **768×1344** | official |
| Template megapixel table includes e.g. 0.4 → 864×480, 0.98 → 1344×768 | template note |
| Duration → frames: **17k+5 @ 24 fps** (math in template) | official + template `ComfyMathExpression` |

**Product defaults (v1):** choose after Phase 0 cost/quality note — **not** auto-port deleted thin worker defaults.  
Candidates: preview-ish 864×480 (0.4 MP 16:9 from template table) vs native 1344×768. Document chosen default in README with rationale (VRAM/time).

Validation: width/height multiples of 32; short edge ≤ 768 and long ≤ 1344 when enforcing “H3 native canvas”; duration in documented min/max (template uses duration float; snap formula lives in graph — product may pass **duration seconds** and let graph snap, or inject precomputed length — pick one and test in Phase 0).

## Product API (v1)

```json
{
  "input": {
    "prompt": "…",
    "width": 1344,
    "height": 768,
    "duration": 5.0,
    "seed": 42
  }
}
```

- No `workflow` enum multi-mode in v1 (single T2V template).  
- No raw graph field.

**Success:** `video_url` or `video` (base64) + meta (`width`, `height`, `seed`, `duration`/`length` if known, `model` id string).

## Delivery

| `BUCKET_*` state | Behavior |
|------------------|----------|
| All four non-empty | URL upload |
| All empty/unset | Inline base64 if size ≤ `MAX_INLINE_VIDEO_BYTES` (default 7e6); else error |
| Any non-empty but incomplete set | **Fail at worker start** (misconfig) |

Never `require_bucket_or_exit` that forces bucket for every deploy.

## Phase plan (execution)

See plan file — four phases only:

1. Pinned Comfy + API workflow + four weights + **headless** MP4+audio smoke  
2. Minimal handler + product inject  
3. S3/base64 + CPU tests  
4. Docker/serverless smoke + VRAM/cold start

## Success (GO)

- Phase 0 headless script produces MP4 with audio via SaveVideo metadata.  
- Serverless `/runsync` product input returns video.  
- CPU unit tests: inject exact nodes; delivery matrix.  
- README: pins, volume layout, canvas default rationale, license.

## Supersedes

`workers/minimax_h3/` deleted; `2026-08-04-*` thin/spike docs historical stubs only.
