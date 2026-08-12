# MiniMax H3 — I2V / first–last frame (Design)

**Дата:** 2026-08-12  
**Статус:** approved (brainstorm)  
**Depends on:** T2V worker + model cache (`workers/minimax_h3_comfy/`, fl2va four weights live)  
**Official:** [MiniMax H3 in ComfyUI](https://docs.comfy.org/tutorials/video/minimax/minimax-h3) — same `MiniMaxH3ImageToVideo` + `fl2va` pack; optional `first_frame` / `last_frame`

## Goal

Product jobs can animate from keyframe images on the **existing** serverless endpoint:

1. **T2V** (unchanged): no image fields → current frozen graph path.  
2. **I2V**: `first_image` → Comfy `first_frame`.  
3. **FL2V**: `first_image` + optional `last_image` → `first_frame` + `last_frame`.

Same four weights, same SaveVideo delivery, same canvas/duration rules.

## Non-goals

- R2V / `ref2va` weights / `MiniMaxH3ReferenceToVideo`  
- Auto canvas from image aspect ratio  
- Explicit `mode` field  
- Raw Comfy workflow escape hatch  
- Model Cache / Docker / pin changes  

## Product API

```json
{
  "input": {
    "prompt": "…",
    "first_image": "https://example.com/start.png",
    "last_image": "data:image/png;base64,…",
    "width": 864,
    "height": 480,
    "duration": 5.0,
    "seed": 42
  }
}
```

| Field | Rule |
|-------|------|
| `prompt` | Required non-empty string |
| `first_image` | Optional. If set → I2V. HTTPS URL **or** raw base64 **or** `data:image/{png,jpeg,jpg,webp};base64,…` |
| `last_image` | Optional. Same encodings as `first_image`. Requires `first_image` |
| `last_image` without `first_image` | **Validation error** |
| No `first_image` | T2V (bit-compatible with current inject) |
| `width` / `height` | Always client-supplied (or defaults 864×480). Same H3 canvas validation as T2V |
| `duration` / `seed` | Unchanged (`seed: -1` → random) |

No `mode` field: presence of `first_image` selects I2V/FL2V.

## Architecture

```text
normalize_input
  → if first_image: resolve_image_ref → stage under COMFY_INPUT_DIR
  → if last_image: same (error if no first)
  → inject_product(…, first_image_name?, last_image_name?)
  → POST /prompt → poll history → SaveVideo path → delivery
  → best-effort cleanup staged files
```

**Approach:** single frozen `workflows/t2va_api.json` + **runtime graph patch** (not a second full workflow file).

- T2V: inject prompt / width / height / duration / seed only; no `first_frame` / `last_frame` keys on node `104`.  
- I2V: add `LoadImage` node `"200"` with `image: <staged basename>`; set `104.inputs.first_frame = ["200", 0]`.  
- FL2V: also add `"201"` + `104.inputs.last_frame = ["201", 0]`.

Node ids `200` / `201` are fixed contract (document in `PINS.md`).

## Image resolve & staging

| Rule | Value |
|------|--------|
| Encodings | `http(s)://…`, raw base64, data URL |
| Formats | PNG, JPEG, WEBP (Pillow + magic / open) |
| Max download / payload | 20 MiB (configurable env optional; default constant) |
| Max pixels | 4096×4096 (16_777_216) default |
| URL scheme | only `http` / `https`; fail closed otherwise |
| Stage dir | `COMFY_INPUT_DIR` (default `/comfyui/input`) |
| Filenames | `{job_id}_first.{ext}`, `{job_id}_last.{ext}` |
| Cleanup | after job (success or fail), best-effort unlink staged files |

RGB conversion before save (PNG) is acceptable so LoadImage always sees a normal file.

## Components

| File | Role |
|------|------|
| `image_input.py` | `resolve_image_ref`, `stage_image`, validation errors |
| `workflow.py` | extend `inject_product` with optional image basenames + LoadImage nodes |
| `handler.py` | wire normalize → stage → inject → cleanup |
| `tests/test_image_input.py` | URL/base64/data URL, last-without-first at normalize level |
| `tests/test_workflow.py` | inject wiring T2V / first / first+last |
| `PINS.md`, `README.md` | API + inject map |
| `scripts/minimax_h3_t2v.py` | `--first-image` / `--last-image` (path or URL; path → base64) |

## Errors

- Validation / resolve failures → `{"error": "<message>"}` (existing handler pattern).  
- Comfy failures unchanged.  
- Partial BUCKET_* still process exit at startup (unchanged).

## DoD

| ID | Check |
|----|--------|
| U1 | Unit: inject T2V has no LoadImage / no first_frame |
| U2 | Unit: inject first only wires `200` → `first_frame` |
| U3 | Unit: inject first+last wires `200`/`201` |
| U4 | Unit: `last_image` without `first_image` rejected |
| U5 | Unit: base64 + data URL decode; oversized rejected |
| L1 | Live: job with `first_image` → COMPLETED → MP4 + audio |
| L2 | Live optional: `first_image` + `last_image` |
| R1 | Live or unit path: T2V without images still works |

## Out of scope follow-ups

- R2V  
- Auto resolution from keyframe aspect  
- Comfy `/upload/image` multipart (direct write to input dir is enough)  
