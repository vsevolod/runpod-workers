# MiniMax H3 — ComfyUI Serverless Worker (Design)

**Дата:** 2026-08-05  
**Статус:** approved direction (replaces thin diffusers ModularPipeline MVP)  
**Связано:** superseded thin worker + Level-1 inject spike (deleted from `workers/`)

## Контекст и решение

Thin-Python путь (diffusers `ModularPipeline` + official ~144 GB pack, или inject
Comfy int8 DiT в stock `MiniMaxH3Transformer3DModel`) **не** даёт целевой
экономии диска/VRAM для **pruned** `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
(~19.5 GB): pruned = **curve AdaLN**, несовместим со stock transformer без порта
архитектуры.

**Решение:** serverless worker на **ComfyUI** (headless), с весами Comfy-Org и
официальными/Comfy workflow-шаблонами MiniMax H3. Product API остаётся простым
(prompt / canvas / duration / seed); внутри — pinned workflow API JSON.

## Goal

1. RunPod Serverless endpoint генерирует **video+audio (t2va / FL2VA)** через
   ComfyUI без ручного UI.
2. Primary DiT: **pruned int8 ConvRot** (~19.5 GB on volume).
3. Стабильный cold start: volume bootstrap, pin Comfy + custom nodes, template
   workflow.
4. Delivery: **S3/R2 URL** (все четыре `BUCKET_*`) или base64 для маленьких
   артефактов (как в других workers монорепо).

## Non-goals (v1)

- Diffusers ModularPipeline / thin inject Level-1.
- Ref2VA / multi-shot / R2V (можно later via second workflow template).
- Произвольный «сырой» `input.workflow` как **единственный** public API
  (внутренний escape hatch OK; product surface — нормализованные поля).
- Bit-exact parity с каждым community template; достаточно стабильного
  smoke + фиксированного pin.
- EU/UK/KR/US serving under Community License without separate authorization
  (ops obligation; document only).

## Architecture

```text
  RunPod /run|/runsync
           │
           ▼
  handler.py  ── validate product input
           │
           ├─ inject → pinned API workflow JSON (node ids fixed)
           │
           ▼
  ComfyUI (localhost:8188)  ── queue_prompt / history / output files
           │
           ▼
  video (+audio) file under Comfy output/
           │
           ▼
  delivery: BUCKET upload → video_url  |  base64 if small + no bucket
```

**Process model (recommended):**

- One container = one GPU worker.
- On start: start ComfyUI server (pinned commit), wait until `/system_stats` OK,
  load models lazily on first job (Comfy default) or optional warm load.
- Handler talks HTTP to Comfy on `127.0.0.1` (same pattern as
  [runpod-workers/worker-comfyui](https://github.com/runpod-workers/worker-comfyui)).

**Why not only Hub `worker-comfyui` image as the product?**

- Stock worker is **image-centric** (`output.images`); MiniMax H3 needs **video**
  (and likely audio mux) collection from Comfy output nodes.
- We need **pinned MiniMax custom nodes** + fixed product schema and monorepo
  tests/deploy path.
- Hub base image / fork of worker-comfyui is still a valid **implementation
  substrate** (Dockerfile FROM or copy handler patterns).

## Model & volume layout

**Network volume (recommended ≥ 80–120 GB** for pruned + TE + VAEs + nodes cache;
minimum ~40 GB if only DiT+minimal extras proven):

```text
/runpod-volume/minimax_h3_comfy/          # COMFY_MODEL_DIR or Comfy models root
  models/
    diffusion_models/
      minimax_h3_fl2va_pruned_int8_convrot.safetensors   # ~19.5 GB primary
    text_encoders/   # Qwen3-VL / whatever workflow requires (Comfy-Org layout)
    vae/
    audio_vae/       # if separate in workflow
    ...
  custom_nodes/      # OR bake nodes into image; volume symlink optional
  workflows/         # exported API JSON templates (also baked in image)
```

**Bootstrap script** (`download_weights.py` / `bootstrap_volume.py`):

- HF: `Comfy-Org/MiniMax-H3` — at least pruned DiT + documented TE/VAE files
  from Comfy-Org README / model card.
- Optional: non-pruned DiT for quality A/B (not default).
- Does **not** download MiniMaxAI ~144 GB official modular pack for v1.

**Comfy paths:** set `COMFYUI_PATH` models via env or symlink
`/comfyui/models` → volume `models/`.

## Product API (v1)

Same spirit as deleted thin worker (client-compatible where possible):

```json
{
  "input": {
    "prompt": "string (required)",
    "width": 864,
    "height": 480,
    "duration": 5.0,
    "seed": -1,
    "workflow": "t2va_pruned_int8"
  }
}
```

| Field | Default | Notes |
|-------|---------|--------|
| `prompt` | — | required, max length TBD (e.g. 8000) |
| `width` / `height` | 864×480 | multiples of 32; bounds from Comfy workflow packing |
| `duration` | 5.0 | map to frames via workflow (document exact snap) |
| `seed` | -1 | random if -1 |
| `workflow` | `t2va_pruned_int8` | template id; only allowlisted ids |

**Success:**

```json
{
  "video_url": "https://…",
  "width": 864,
  "height": 480,
  "length": 124,
  "seed": 42,
  "requested_duration": 5.0,
  "output_duration": 5.166…,
  "fps": 24,
  "model": "Comfy-Org/MiniMax-H3/pruned_int8",
  "workflow": "t2va_pruned_int8"
}
```

Or `video` base64 when no bucket and size ≤ inline limit.

**Errors:** `{ "error": "…" }`; OOM / Comfy crash → `refresh_worker: true`.

**Escape hatch (internal / debug only, not default product):**
`input.comfy_workflow` full API graph — gated by env `ALLOW_RAW_WORKFLOW=1`.

## Pins (must freeze in plan execution)

| Component | Pin policy |
|-----------|------------|
| ComfyUI | git SHA or release tag |
| MiniMax H3 custom nodes | repo + SHA (from Comfy-Org / community package used by official workflows) |
| worker base | optional: `runpod/worker-comfyui:<ver>-base` or CUDA image + install |
| Primary weights | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` filename exact |
| Workflow template | file under `workflows/t2va_pruned_int8_api.json` + node-id map for inject |

## GPU / ops

| | Recommendation |
|--|----------------|
| GPU | Start **48 GB** class for smoke; measure peak; may fit **24–40 GB** with pruned int8 — **validate empirically** |
| Workers | 1 GPU / worker; FlashBoot-friendly long load |
| Volume | Network Volume mounted `/runpod-volume` |
| Cold start | Comfy process start + first model load dominates |

## License

MiniMax H3 Community License still applies to model weights (Excluded Territories,
downstream restrictions). Ship license text + NOTICE. ComfyUI / nodes have their
own licenses — list in NOTICE.

## Testing strategy

| Layer | What |
|-------|------|
| Unit | request normalize, workflow inject (node fields), delivery mode |
| Integration (CPU) | mock Comfy HTTP; handler returns planned delivery |
| GPU smoke | Pod or serverless: one job `test_input.json` → MP4 finite + length |
| Regression | pin workflow JSON hash; fail if template missing nodes |

## Success criteria (GO)

1. Volume bootstrap docs + script install pruned DiT + required companions.
2. Serverless job with product input returns video (URL or base64).
3. Default path uses **pruned int8** only (no 144 GB official pack).
4. CPU unit tests green in monorepo.
5. README: volume size, VRAM, env, license, smoke steps.

## Supersedes

| Old | Status |
|-----|--------|
| `workers/minimax_h3/` thin ModularPipeline + Level-1 inject | **Deleted** 2026-08-05 |
| Specs/plans `2026-08-04-minimax-h3-t2v-worker*` / `*-level1-comfy-dit-spike*` | **Historical only** — do not implement |
