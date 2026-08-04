# MiniMax H3 T2V RunPod serverless worker

**Дата:** 2026-08-04  
**Статус:** approved v3 (+ canvas/bucket/deps clarifications for plan rev 2)  
**Plan:** `docs/superpowers/plans/2026-08-04-minimax-h3-t2v-worker.md` (rev 2 — not yet execution-approved)  
**Backend (зафиксирован):** **B — Hugging Face diffusers ModularPipeline**  
**Официальный upstream / веса:** [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)  
**Лицензия:** [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)  
**Comfy template (контекст, не runtime):** [video_minimax_h3_t2v.json](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json)  
**diffusers docs:** [minimax_h3.md](https://github.com/huggingface/diffusers/blob/minimax-h3/docs/source/en/api/pipelines/minimax_h3.md)

## Контекст

Monorepo `runpod-workers` — thin serverless workers (`krea2`, `joycaption`,
`lora_downloader`): Python handler, веса на Network Volume, без ComfyUI runtime.

Нужен worker **MiniMax H3 text-to-video (T2V / t2va)**: prompt → MP4 с **native
stereo audio** (joint generation, не пост-продакшен).

| Режим | Checkpoint half | MVP |
|-------|-----------------|-----|
| T2V / t2va | FL2VA / `transformer/` | **yes** |
| I2V / fl2va (first/last frame) | тот же half | phase 2 |
| R2V / ref2va | `transformer_ref/` | phase 3 |

Comfy templates полезны как product reference (режимы, duration grid, canvas),
но **не** задают weight layout и **не** задают sampling profile для
diffusers/SGLang.

## Цели (MVP)

- Thin RunPod worker: `workers/minimax_h3/`.
- **T2V only:** `prompt` + resolution + duration + seed → MP4 (video + audio muxed).
- Backend: **diffusers ModularPipeline** (`MiniMaxH3Blocks` / t2va workflow).
- Веса: **MiniMaxAI/MiniMax-H3** layout, совместимый с diffusers (не Comfy-Org pack).
- Network Volume bootstrap: `download_weights.py` качает **только** то, что нужно
  backend B для t2va (без `transformer_ref/`, без Comfy quant files).
- Load once at process start; fail-fast if volume incomplete.
- Delivery: **S3/bucket via `BUCKET_ENDPOINT_URL`** (как krea2) + base64 fallback
  только для мелких smoke-роликов.
- License compliance documented; **deploy только в Allowed Territory DCs**.

## Не входит в scope (MVP)

- I2V frames, R2V / `transformer_ref`.
- H3-Context-IR, H3-Regenerate-2K.
- Cloud MiniMax API (`api_minimax_h3_*`).
- ComfyUI runtime (D) и thin port Comfy quant nodes (C) — **не MVP**; C только
  если B не проходит smoke (escalation path, отдельный rev спеки + другой
  download pack).
- SGLang (A) — fallback multi-GPU path, не default; отдельный pack/`MODEL_DIR`
  layout при переключении.
- Client-tunable `steps` / Comfy `res_multistep` parity.
- LoRA, multi-prompt batch, bake weights in image.
- Клиентский `scripts/minimax_h3_*.py` — after smoke.

## Подход

### Backend: B (diffusers) — locked for MVP

| | |
|--|--|
| **Choice** | Hugging Face **diffusers** ModularPipeline for MiniMax-H3 |
| **Why** | Thin Python like krea2; official `t2va` path; documented **1×80 GB + CPU offload**; no Comfy graph engine |
| **Install note** | Day-0: install from PR until merged (`pip install git+https://github.com/huggingface/diffusers.git@refs/pull/14355/head` or successor) — pin SHA in Dockerfile |
| **Fallback** | If B smoke fails on target GPU: escalate to **A (SGLang)** with multi-GPU; requires **new** download layout spike and API sampling note — not silent swap |
| **Not MVP** | C (Comfy quant thin port), D (headless Comfy) |

Comfy-Org single-file quants (**~40 GB**) нужны **только** варианту C/D. Для B они
**несовместимы** как primary load path. Downloader MVP **не** качает Comfy pack.

### Weight layout ↔ backend (invariant)

| Backend | Weight source | Layout |
|---------|---------------|--------|
| **B diffusers (MVP)** | `MiniMaxAI/MiniMax-H3` | root: `transformer/`, `text_encoder/`, `vae/`, `audio_vae/`, tokenizers/processors/schedulers, `modular_model_index.json` — **без** `transformer_ref/` for T2V |
| A SGLang | `MiniMaxAI/MiniMax-H3` | often `FL2VA/*` (or cookbook path); **not** identical to flat diffusers subfolders alone |
| C/D Comfy | `Comfy-Org/MiniMax-H3` | `diffusion_models/`, `text_encoders/`, `vae/` single files |

**Правило:** выбор backend **до** реализации downloader. Один default pack =
один backend. Нет «качаем Comfy на всякий случай».

### Sampling profile (backend-native, not Comfy)

Comfy template uses `res_multistep` + `simple` + **20** steps. That is **not**
the official SGLang/diffusers profile and **must not** be exposed as a portable
`steps` API.

**MVP sampling — fixed inside pipeline (not client input):**

| Parameter | MVP value | Source |
|-----------|-----------|--------|
| fps | **24** | model / diffusers docs |
| duration window | **requested ∈ [5, 14.375] s** | last legal 17n+5 grid point ≤15s output is 345 frames = 14.375s (see Frame length) |
| frame snap | `17*n + 5` | video VAE grid |
| video scheduler shift | **12.0** | checkpoint / docs (`scheduler`) |
| audio scheduler shift | **3.0** | checkpoint / docs (`audio_scheduler`) |
| `num_inference_steps` | **50** (fixed in pipeline; not client input) | diffusers ModularPipeline default on minimax-h3 branch |
| guidance | none (CFG-distilled) | no negative prompt / guidance_scale |
| client `steps` | **removed from API** | avoids false Comfy parity |

Note (diffusers): `num_inference_steps` **counts the terminal zero** in the sigma
grid → one fewer model eval than a naive “N denoise steps” reading. README may
mention this; do **not** re-export as user `steps`.

**Comfy parity is out of MVP.** Frame-length formula may still match Comfy’s
17k+5 grid (shared VAE constraint); quality/sampling will not match 20-step
Comfy template.

### Frame length formula

Model grid: `num_frames = 17*n + 5` at 24 fps. Diffusers validates that the
resulting duration stays within the generation window; the **last legal grid
point with `frames/24 ≤ 15`** is:

| frames | duration |
|-------:|---------:|
| **345** (= 17×20 + 5) | **14.375 s** |
| 362 (= 17×21 + 5) | 15.083… s — **illegal** |

Therefore the API accepts **requested** `duration ∈ [5, 14.375]` only. After
snap, every accepted request stays legal — **no second reject-after-snap** for
the upper bound.

```python
MIN_DURATION_SEC = 5.0
MAX_DURATION_SEC = 14.375  # 345 / 24; max legal requested duration
FPS = 24

def snap_num_frames(duration_sec: float) -> int:
    """Map requested duration to model frame count at 24 fps (17n+5 grid)."""
    frames = max(5, round(duration_sec * 24))
    frames = frames + (5 - (frames % 17)) % 17
    return frames
```

Validation order:

1. Reject if `duration` not in **[5, 14.375]** (inclusive).
2. `length = snap_num_frames(duration)` (for `duration ≤ 14.375`, snap stays ≤ 345).
3. No separate “output > 15s” check required for the upper bound.

Golden values:

| duration_sec (requested) | raw `round(d*24)` | snapped `length` | output_duration `length/24` |
|-------------------------:|------------------:|-----------------:|----------------------------:|
| 5 | 120 | 124 | 5.166… s |
| 6 | 144 | 158 | 6.583… s |
| 8 | 192 | 192 | 8.0 s |
| 10 | 240 | 243 | 10.125 s |
| 14.375 | 345 | 345 | 14.375 s |
| 15 | — | — | **schema reject** (outside API range) |

Align with diffusers MiniMax-H3 pre-encoder validation
(`before_encoder.py` on minimax-h3 branch).

### Resolution (MVP)

- Client: `width`, `height` (int) — pixel **W×H**, both multiples of **32**.
- Default: **864 × 480** (preview / speed; area under released cap).
- **Backend packing parity** (diffusers `packing.py` @
  `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`), **not** “max side 1344”:

  | Constant | Value |
  |----------|------:|
  | short edge (nominal) | 768 |
  | max area | `768 * 1344` (= 1_032_192) |
  | multiple | 32 |
  | aspect | **1:4 … 4:1** |

  Upstream `resolve_canvas_size(aspect_w, aspect_h)` builds a ratio-specific
  canvas (short edge 768, area cap, round to 32). Examples:

  | Ratio | height × width |
  |-------|----------------|
  | 16:9 | 768 × 1344 |
  | 4:1 | **512 × 2016** (long side **> 1344** — valid) |
  | 1:4 | **2016 × 512** |

  Worker validation for explicit `width`/`height`:

  1. multiples of 32; aspect ∈ [1/4, 4];
  2. `nom_h, nom_w = resolve_canvas_size(width, height)` (ratio only);
  3. accept iff `width * height <= nom_h * nom_w` (previews smaller than
     nominal OK; oversize rejected).

  Do **not** reject solely because `max(width,height) > 1344`.
- No `aspect_ratio` + megapixels helper in MVP.

### GPU

- **Target MVP:** **1 × 80 GB** + diffusers `ComponentsManager` auto CPU offload
  (docs: `enable_auto_cpu_offload`, reserve margin ~12 GB) + optional flash attn backend.
- Not claiming 24 GB without int8/group-offload recipe (documented as optional
  later, not MVP default).
- Host RAM: large (full bf16 TE+DiT); volume + machine RAM sized accordingly.

## Layout

```text
workers/minimax_h3/
├── Dockerfile
├── handler.py                 # RunPod entry; load once; T2V; upload policy
├── schemas.py
├── download_weights.py        # MiniMaxAI pack for backend B only
├── requirements.txt
├── test_input.json
├── NOTICE
├── LICENSES/
│   ├── MINIMAX-H3-COMMUNITY.txt   # FULL license text (required; no pointer-only)
│   └── RUNPOD-WORKER-SDXL-MIT.txt # if patterns reused
├── README.md                  # deploy + License + DC/user geo + payload
├── h3_infer/
│   ├── __init__.py
│   ├── duration.py            # snap_num_frames + [5, 14.375] validation
│   ├── request.py             # normalize beyond schema; canvas checks
│   └── pipeline.py            # load + generate_t2v (diffusers)
└── tests/
    ├── test_duration.py
    ├── test_request.py
    ├── test_download_weights.py
    ├── test_response_payload.py   # size guard / delivery mode selection
    └── test_schemas.py
```

Mux via `diffusers.utils.export_utils.encode_video` (or thin helper inside
`pipeline.py` / `handler.py`). No separate `encode.py` module.

## Ресурсы (disk) — backend B

T2V diffusers layout on disk is about **144 GB / ~134 GiB** (full t2va half without
`transformer_ref/`). Approximate breakdown:

| Component | ~Size |
|-----------|------:|
| `transformer/` (FL2VA DiT) | ~62 GB |
| `text_encoder/` (Qwen3-VL-32B) | ~62 GB |
| VAEs + processor/tokenizer/schedulers + indexes | ~remainder → **~144 GB total** |
| `transformer_ref/` | **do not download** for T2V MVP |

Network volume: **≥ 200 GB** (weights + headroom + cache crumbs) — correct for
~144 GB pack. Separate volume from krea2/joycaption: **yes**.

## Volume layout (diffusers / MiniMaxAI — default)

```text
/runpod-volume/minimax_h3/          # MODEL_DIR
  modular_model_index.json          # and/or model_index.json as required
  transformer/                      # t2va + fl2va DiT
  text_encoder/
  vae/
  audio_vae/
  tokenizer/   processor/           # as present in repo
  scheduler/   audio_scheduler/     # shifts 12.0 / 3.0 in config
  # NOT present for T2V MVP:
  # transformer_ref/
  # FL2VA/ Ref2VA/  — only if a future pack mode needs original trees
```

Exact `allow_patterns` for `snapshot_download` fixed at implement against current
HF tree so that `ModularPipeline.from_pretrained(MODEL_DIR, local_files_only=True)`
loads without network.

Bootstrap:

```bash
python workers/minimax_h3/download_weights.py --output /runpod-volume/minimax_h3
```

Then endpoint: `MODEL_DIR=/runpod-volume/minimax_h3`, `LOCAL_FILES_ONLY=1`.

## Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_DIR` | `/runpod-volume/minimax_h3` | Root for MiniMaxAI/diffusers layout |
| `HF_HOME` | unset | Optional during bootstrap |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | unset | Download auth / rate limits |
| `LOCAL_FILES_ONLY` | recommend `1` after warm | Explicit `local_files_only=` into from_pretrained |
| `LOG_LEVEL` | `INFO` | |
| `MAX_DURATION_SEC` | `14.375` | Upper bound for **requested** duration (= 345/24) |
| `MIN_DURATION_SEC` | `5` | Lower bound for **requested** duration |
| `BUCKET_ENDPOINT_URL` | unset | S3-compatible endpoint (required with the three below) |
| `BUCKET_ACCESS_KEY_ID` | unset | Required for real S3 client (runpod 1.7.9) |
| `BUCKET_SECRET_ACCESS_KEY` | unset | Required for real S3 client |
| `BUCKET_NAME` | unset | **Required** explicit bucket; without it helper uses `%m-%y` and may not create it |
| `MAX_INLINE_VIDEO_BYTES` | `7_000_000` | Max **raw MP4** bytes for inline base64 |
| `REQUIRE_BUCKET` | `0` | If `1`, refuse start unless **all four** `BUCKET_*` are set |
| `HF_XET_HIGH_PERFORMANCE` | unset / `1` during bootstrap | Hub xet; **not** `hf_transfer` |

Bucket is “configured” only when **all four** env vars are non-empty. Endpoint
alone is **not** enough: missing keys → `upload_file_to_bucket` writes
`local_upload/` and returns a **local path** (must not be returned as
`video_url`). Always pass `bucket_name=os.environ["BUCKET_NAME"]` and
`extra_args={"ContentType": "video/mp4"}`.

Canvas: packing `resolve_canvas_size` + area ≤ nominal for aspect (see Resolution).

No client `steps` env; pipeline constant `num_inference_steps=50`.

## API

### Input (`job.input`) — T2V

| Field | Type | Required | Default | Notes |
|-------|------|:--------:|---------|-------|
| `prompt` | `str` | **yes** | — | Non-empty after strip; max ~8000 chars |
| `width` | `int` | no | `864` | Multiple of 32 |
| `height` | `int` | no | `480` | Multiple of 32 |
| `duration` | `number` | no | `5` | **Requested** seconds; **[5, 14.375]** inclusive; then snap to frames |
| `seed` | `int` | no | random | Non-negative; omit → worker samples and returns used seed |

**Removed from MVP:** `steps` (fixed **50** inside pipeline), `fps` (fixed 24),
frames/images, CFG, negative prompt.

Example:

```json
{
  "input": {
    "prompt": "Single continuous shot, about 5 seconds. A red fox runs through snow. Soft wind SFX, no music.",
    "width": 864,
    "height": 480,
    "duration": 5,
    "seed": 42
  }
}
```

### Success output

Always return geometry/timing metadata. Video delivery is **either** URL **or**
inline base64 — never both required; prefer URL when bucket configured.

**With bucket (`BUCKET_ENDPOINT_URL` set):**

```json
{
  "video_url": "https://…/….mp4",
  "seed": 42,
  "width": 864,
  "height": 480,
  "requested_duration": 5,
  "length": 124,
  "fps": 24,
  "output_duration": 5.166666666666667,
  "model": "MiniMaxAI/MiniMax-H3"
}
```

**Inline fallback (no bucket, and raw MP4 ≤ `MAX_INLINE_VIDEO_BYTES`):**

```json
{
  "video": "<raw base64 mp4, no data: prefix>",
  "seed": 42,
  "width": 864,
  "height": 480,
  "requested_duration": 5,
  "length": 124,
  "fps": 24,
  "output_duration": 5.166666666666667,
  "model": "MiniMaxAI/MiniMax-H3"
}
```

Field rules:

| Field | Meaning |
|-------|---------|
| `requested_duration` | Client `duration` after schema normalize |
| `length` | Frame count after snap (`snap_num_frames`) |
| `fps` | Always `24` |
| `output_duration` | `length / fps` (actual timeline length; **≠** requested when snap expands) |
| `video_url` / `video` | Exactly one present on success |

**Never** return a lone `duration: 5` when `length` is 124 — that was wrong in v1.

### Payload / delivery policy (P1 fix)

RunPod limits ([docs](https://docs.runpod.io/serverless/workers/handler-functions#payload-limits)):

| Endpoint | Limit |
|----------|------:|
| `/run` | **10 MB** |
| `/runsync` | **20 MB** |

Base64 expands ≈ 4/3. Guard must use **raw MP4 size**, not a loose 8 MB that
becomes ~10.67 MB base64.

**Policy (bucket-first; runpod 1.7.9 `upload_file_to_bucket`):**

1. Write MP4 to temp path under `/{job_id}/output.mp4`.
2. `raw_size = path.stat().st_size` (do **not** `read_bytes()` until base64 branch).
3. If **all four** bucket env vars set →
   `rp_upload.upload_file_to_bucket(file_name=..., file_location=..., bucket_name=BUCKET_NAME, extra_args={"ContentType": "video/mp4"})`.
   Require returned value starts with `http://` or `https://` (reject
   `local_upload/...` paths). Return `video_url`.
4. Else if `raw_size <= MAX_INLINE_VIDEO_BYTES` → `read_bytes()` + base64 → `video`.
5. Else → error asking to configure full `BUCKET_*` set or reduce size.
6. Production: `REQUIRE_BUCKET=1` (start fail if any of four missing).
7. CUDA OOM / unexpected generate failure: include `"refresh_worker": true` (krea2/joycaption).
8. Cleanup temp dir (`rp_cleanup`).

Unit tests: `bucket_configured`, delivery mode, non-URL upload rejection.

### Error output

```json
{ "error": "human-readable message" }
```

No stack traces to client.

## Inference flow

1. Validate schema + `request.normalize`.
2. `requested_duration` ∈ **[5, 14.375]**; `length = snap_num_frames(...)`
   (no second upper-bound reject).
3. Canvas validate (packing `resolve_canvas_size` + area ≤ nominal).
4. Resolve seed.
5. `pipeline.generate_t2v(...)` with **`num_inference_steps=50`**.
6. Mux via `encode_video` (**requires av**).
7. Delivery: `st_size` → bucket (4 env) / base64 / error; reject non-URL uploads.
8. OOM → `{ "error": "...", "refresh_worker": true }`; other failures may set
   `refresh_worker` too (krea2 pattern).

## download_weights.py

### Purpose

Bootstrap volume for **backend B (diffusers)** only. CPU Pod friendly; no torch.

### CLI

```bash
python download_weights.py --output /runpod-volume/minimax_h3
python download_weights.py --output /runpod-volume/minimax_h3 --dry-run
python download_weights.py --output /runpod-volume/minimax_h3 --repo MiniMaxAI/MiniMax-H3
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--output` | `MODEL_DIR` or `./models/minimax_h3` | Local root |
| `--repo` | `MiniMaxAI/MiniMax-H3` | HF repo |
| `--token` | `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Auth |
| `--hf-home` | `HF_HOME` or empty | Cache root for this process |
| `--dry-run` | false | Print patterns / planned paths, no download |

**No** `--pack comfy`, **no** `--include-ref2va` in T2V MVP (R2V later).

### What to download (T2V)

`snapshot_download` with **allow_patterns only** (no long ignore list): indexes +
`transformer/`, `text_encoder/`, `vae/`, `audio_vae/`, tokenizer, processor,
scheduler, audio_scheduler. Patterns do not match `transformer_ref/` or
`Ref2VA/` → those are not fetched.

Idempotent; exit `0` / `1` / `2` as krea2-style; print suggested env after success.

### Disk warning

Script should print approximate size (**~144 GB / ~134 GiB**) and refuse to
claim a “small pack”.

## Model loading

```text
Process start
  → MODEL_DIR exists + modular index present (fail-fast)
  → ModularPipeline.from_pretrained(MODEL_DIR, local_files_only=…)
  → load_components(dtype=bfloat16)
  → ComponentsManager auto CPU offload on 1×80GB (or documented recipe)
  → ready
```

## Dockerfile

- CUDA base aligned with monorepo (or newer if diffusers/torch require).
- Python 3.12; torch CUDA wheel; **diffusers from pinned git SHA** until release;
  transformers, accelerate, etc. as required by MiniMax-H3 docs.
- Copy `workers/minimax_h3/` → `/app`.
- No weight download at build.
- `CMD ["python", "-u", "/app/handler.py"]`.
- RunPod: Dockerfile `workers/minimax_h3/Dockerfile`, context **repo root**.

## Dependencies (runtime)

| Package | Notes |
|---------|--------|
| `runpod` | `>=1.7.9` — handler + `rp_upload.upload_file_to_bucket` / cleanup |
| `torch` | CUDA |
| `diffusers` | pinned git SHA (PR 14355 or merged release) |
| `transformers` | Qwen3-VL load path |
| `accelerate` | device / offload |
| `huggingface-hub` | download; xet / `HF_XET_HIGH_PERFORMANCE`; no `hf_transfer` |
| `safetensors`, `numpy` | |
| **`av` (PyAV)** | **required** — `encode_video` ImportError without it |
| `diffusers` | git pin `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc` (PR #14355) |

## Error handling

| Case | Response |
|------|----------|
| Schema / empty prompt | `{ "error": "..." }` |
| duration outside [5, 14.375] | `{ "error": "..." }` |
| width/height not % 32 or canvas/aspect illegal | `{ "error": "..." }` |
| Inline too large, no bucket | `{ "error": "video too large for inline..." }` |
| `REQUIRE_BUCKET=1` but no bucket | start fail or job error (prefer start fail) |
| CUDA OOM | `{ "error": "..." }` + logs |
| Missing weights | process start fail |

## Testing

| Layer | Coverage |
|-------|----------|
| Unit | duration snap + reject outside [5, 14.375]; schema; download patterns; delivery mode; canvas bounds |
| Smoke GPU | 864×480, duration 5 → length 124; MP4+audio; prefer bucket |
| CI | unit only |

## License / NOTICE (P0 — release blocker)

Source: [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
(License date August 2, 2026). This section is an engineering checklist, **not
legal advice**. Operators must use the **full Agreement text**.

### Territorial restriction (infrastructure **and** users)

- **Applicable Territory:** worldwide **excluding Excluded Territories**.
- **Excluded Territories:** **European Union, United Kingdom, Republic of Korea,
  United States of America**.
- Use, reproduction, distribution, **running**, or display of MiniMax H3 Works
  (including via **Hosted Services**) **outside Applicable Territory is not
  authorized**. Separate written license from MiniMax required for those regions.

**Two layers — both mandatory:**

| Layer | Requirement |
|-------|-------------|
| **Infrastructure** | RunPod endpoint, Network Volume, and worker pods **must not** be in DCs in US / EU / UK / KR under Community License. |
| **Users / access** | The **product** that exposes this endpoint **must not** provide the service to users in Excluded Territories (geo-blocking, account region, IP policy, or equivalent access control). DC placement alone is **insufficient**. |

If product needs those markets → obtain MiniMax authorization first; do not
deploy or open access under Community License alone.

### Hosted service obligations (RunPod serverless = Hosted Service)

Before offering the endpoint to third parties / end users, Licensee must:

1. Bind each recipient/user to enforceable terms **at least as protective** as
   Section V + Exhibit A (AUP).
2. **Provide users a copy of the MiniMax H3 Community License Agreement (full
   text)** — not a summary-only or link-only substitute when the license requires
   providing a copy of the Agreement (distribution / Hosted Service obligations).
   Product UX may also deep-link to HF, but the **full text** must be available
   to users (e.g. served from product legal pages and mirrored in
   `LICENSES/MINIMAX-H3-COMMUNITY.txt` in this repo for operators).
3. Implement **reasonable safeguards** against prohibited uses/outputs; maintain
   and periodically review them.
4. Provide a **reasonably accessible reporting mechanism** for suspected
   violations; investigate good-faith reports; mitigate / remove / suspend as
   appropriate.
5. Not use Outputs to improve other AI models (except MiniMax H3 / derivatives).
6. Comply with Acceptable Use Policy (Exhibit A), including no military use, etc.

### Commercial / attribution

- Yearly revenue **> USD 20M** → prior written authorization (`api@minimax.io`,
  subject line per license).
- Commercial UI must **prominently display “MiniMax H3”**.
- NOTICE file for distributions of the Works; encourage “Powered by MiniMax H3”.
- Encoder stack includes **Qwen3-VL-32B (Apache-2.0)** — note in NOTICE.

### Repo artifacts

- `LICENSES/MINIMAX-H3-COMMUNITY.txt` — **full** license text (verbatim). **No
  pointer-only** file.
- `NOTICE` — MiniMax H3 notice string + Qwen + RunPod pattern attributions.
- README **License** section: Excluded Territories (DC **and** users), hosted
  obligations including **user copy of Agreement**, commercial threshold, link
  to HF LICENSE for convenience (full text still in repo).

## Deploy checklist

0. **License / DC:** Infrastructure region is **outside** US, EU, UK, KR. If
   not — **stop**.
1. **License / users:** Product-side **geo or access restriction** prevents
   offering the Hosted Service to users in Excluded Territories. Documented and
   enabled before public traffic.
2. **License / Agreement copy:** End users receive / can access the **full**
   MiniMax H3 Community License Agreement text (not summary-only).
3. Hosted-service: ToS/AUP ≥ Section V + Exhibit A; safeguards plan; reporting
   channel live.
4. Network Volume **≥ 200 GB** in an **allowed** DC; same DC as endpoint.
5. Pod + mount; `download_weights.py --output /runpod-volume/minimax_h3`
   (~144 GB pack).
6. Verify `transformer/`, `text_encoder/`, VAEs present; no dependency on
   `transformer_ref` for T2V.
7. Serverless Git: Dockerfile `workers/minimax_h3/Dockerfile`, context repo root.
8. GPU: **1×80 GB** class (or better); env `MODEL_DIR`, `LOCAL_FILES_ONLY=1`.
9. **Bucket:** all four of `BUCKET_ENDPOINT_URL`, `BUCKET_ACCESS_KEY_ID`,
   `BUCKET_SECRET_ACCESS_KEY`, `BUCKET_NAME`; `REQUIRE_BUCKET=1` for prod;
   `upload_file_to_bucket(..., bucket_name=..., ContentType=video/mp4)`.
10. Smoke `test_input.json` (864×480, duration 5); `video_url`; audio;
    `length=124`; `output_duration≈5.167`.
11. README: payload limits; territory (DC+users); Agreement copy; sampling
    (`num_inference_steps=50`, not Comfy 20).

## Phase plan

| Phase | Deliverable |
|-------|-------------|
| **0** | Spec approved (this v2) |
| **1** | `download_weights.py` (MiniMaxAI/diffusers pack) + unit tests |
| **2** | Handler + diffusers pipeline T2V + bucket delivery + smoke |
| **3** | I2V optional frames (same transformer half) |
| **4** | R2V + `transformer_ref` download flag |
| **5** | Client script; optional int8/group-offload recipe for smaller GPUs |

## Decisions log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backend MVP | **B diffusers** | Thin worker; 1×80GB offload docs; official ModularPipeline |
| Fallback | A SGLang only after B fail; new pack spike | Different serve layout |
| Comfy pack download | **Not in MVP** | Only for C/D; wastes ~40 GB and wrong layout for B |
| Sampling API | **No client `steps`**; pipeline **`num_inference_steps=50`** | diffusers ModularPipeline default; not Comfy 20 |
| Frame grid | 17n+5 snap | Shared VAE constraint; not full Comfy parity |
| Duration range | **Requested [5, 14.375]** | Max legal grid ≤15s output is 345 frames; no dual reject |
| Timing fields | `requested_duration` + `length` + `fps` + `output_duration` | Avoid lying `duration: 5` with 124 frames |
| Canvas | `resolve_canvas_size` + area ≤ nominal; allow 2016×512 | packing.py parity; not max-side 1344 |
| Video delivery | Four `BUCKET_*` + `upload_file_to_bucket(bucket_name=)`; `st_size`; inline ≤7 MB; reject local paths | runpod 1.7.9; no fake video_url |
| OOM | `refresh_worker: true` | krea2/joycaption |
| deps | av + diffusers `@abc5e9bf…` | encode_video + PR #14355 |
| Download speed | Hub xet / `HF_XET_HIGH_PERFORMANCE`; no `hf_transfer` | `hf_transfer` deprecated |
| Disk estimate | **~144 GB** T2V half; volume ≥200 GB | Measured pack class |
| GPU | 1×80 GB + offload | Diffusers documented recipe |
| License territory | **Block US/EU/UK/KR for DC and end users** | Community License — release blocker |
| License text | Full Agreement in repo + provided to users | No pointer-only |
| Hosted obligations | ToS/AUP, safeguards, reporting, Agreement copy | License §V / distribution |
| Worker path | `workers/minimax_h3/` | Room for I2V later |

## Open questions (residual)

1. Product-side ownership of ToS text, user geo policy implementation, reporting
   URL, and serving the full Agreement to end users (engineering states
   requirements; product supplies copy and enforcement).

(`num_inference_steps` and upload API are **decided** — not open.)

## Success criteria

- [ ] Spec v3 reviewed; P0 license (DC **+ users** + full Agreement) accepted.
- [ ] Downloader: MiniMaxAI/diffusers T2V half only (~144 GB class).
- [ ] Unit tests: duration [5, 14.375]; delivery mode; schema (no `steps`).
- [ ] Smoke: MP4+audio; `length`/`output_duration`; bucket URL in prod.
- [ ] README: territory (infra+users), Agreement copy, payload, steps=50 fixed.
- [ ] No weights in git.

## References

- MiniMaxAI model card: https://huggingface.co/MiniMaxAI/MiniMax-H3  
- License: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE  
- diffusers MiniMax-H3: https://github.com/huggingface/diffusers/blob/minimax-h3/docs/source/en/api/pipelines/minimax_h3.md  
- Duration / pre-encoder validation: `diffusers/.../minimax_h3/before_encoder.py` (minimax-h3 branch)  
- Canvas packing: `packing.py` @ `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc` (`resolve_canvas_size`)  
- Default `num_inference_steps=50`: modular pipeline utils on same SHA  
- RunPod upload: `runpod-python` 1.7.9 `rp_upload.py` (`upload_file_to_bucket`, `get_boto_client`)  
- SGLang cookbook (fallback A): https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3  
- RunPod payload limits: https://docs.runpod.io/serverless/workers/handler-functions#payload-limits  
- Hub env (`HF_XET_HIGH_PERFORMANCE`; `hf_transfer` deprecated): https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables  
- Comfy tutorial (product reference only): https://docs.comfy.org/tutorials/video/minimax/minimax-h3  
- krea2 bucket env: `workers/krea2/handler.py`; this worker uses `upload_file_to_bucket` for MP4  
- Monorepo peer: `docs/superpowers/specs/2026-07-26-joycaption-worker-design.md`
