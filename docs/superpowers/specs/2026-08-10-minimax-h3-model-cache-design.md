# MiniMax H3 — RunPod Model Cache (вместо своего Network Volume)

**Дата:** 2026-08-10 (rev 4 — V4.1 required with V2; failure-signature fix)  
**Статус:** implementation landed (V0/V1 local green); G0 + V2/V4.1 RunPod still required for DoD  
**Worker:** `workers/minimax_h3_comfy/`  
**Parent design:** [`2026-08-05-minimax-h3-comfyui-serverless-design.md`](./2026-08-05-minimax-h3-comfyui-serverless-design.md)  
**Refs:**
- [RunPod: Use Hugging Face models (cached)](https://docs.runpod.io/serverless/development/huggingface-models)
- [RunPod: Cached models](https://docs.runpod.io/serverless/endpoints/model-caching)
- [RunPod tutorial: model caching text](https://docs.runpod.io/tutorials/serverless/model-caching-text)
- [runpod-workers/model-store-cache-example](https://github.com/runpod-workers/model-store-cache-example)
- HF: [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)  
- HF API blobs (size proof): `https://huggingface.co/api/models/Comfy-Org/MiniMax-H3?blobs=true`

## Review findings addressed

### Rev 2

| ID | Finding | Spec change |
|----|---------|-------------|
| P1 | Full HF repo ~465 GB makes cache path unproven | **Pre-implementation gate G0** + **primary design: slim HF repo** (~42.5 GB, four files only). Full `Comfy-Org/MiniMax-H3` is not the default production Model ID. |
| P1 | `ALLOW_HF_DOWNLOAD` vs 45 GB container disk | **Removed from v1.** Runtime download of four weights (~42.47 GB) cannot fit image + temps + video on 45 GB. Fallback = **legacy Network Volume only**. |
| P1 | Divergent model identity (`HF_MODEL_ID`, `HF_SNAPSHOT`, UI pin `:hash`) | **Single id: `MODEL_NAME`** (RunPod convention), default slim or org/name without pin. v1 resolves **`refs/main` only** (or exactly one snapshot). Pin/hash syntax deferred until live format is verified. |
| P2 | Lexicographic first snapshot is arbitrary | **Removed.** `refs/main` → else **exactly one** snapshot dir → else **fail** listing candidates. |
| P2 | Python ↔ `start.sh` interface unclear | **Contract:** `python /app/model_store.py` always materializes **`/models`**; shell only verifies `/models` + replace Comfy models link. No JSON/env parsing in shell. |
| P3 | pytest vs unittest; `du` through symlinks | Tests = **`unittest`** (existing suite style). Weight size logs use **`du -hL`** (follow links). |

### Rev 3

| ID | Finding | Spec change |
|----|---------|-------------|
| P1 | DoD allowed “V2 cache **or** V4.1 volume-only” | **Migration done requires V2.1–V2.7 (cache path).** V4.1 is not a substitute for V2. |
| P2 | `Dockerfile.bootcheck` omitted | Must **COPY `model_store.py`** (and keep smoke working after `start.sh` change). |
| P2 | `/models` both fixed and configurable | **No `COMFY_MODELS_ROOT` in v1.** Hard-coded **`/models`** only. |
| P2 | `ln -sfn` alone fails if `/comfyui/models` is a real directory | Keep **`rm -rf "${COMFYUI_PATH}/models"`** then **`ln -sfn /models "${COMFYUI_PATH}/models"`** (existing start.sh pattern). |

### Rev 4

| ID | Finding | Spec change |
|----|---------|-------------|
| P1 | Goal keeps legacy volume fallback, but DoD treated V4.1 as optional | **V4.1 required** in addition to V2 (prove volume fallback still works). **V4.2 optional.** |
| P3 | Failure text “full without expected paths” | Full `Comfy-Org/MiniMax-H3` **does** contain the four T2V paths; drop that misstatement. |

## Контекст

v1 worker bootstraps weights from a **user Network Volume**:

```text
MODEL_DIR=/runpod-volume/minimax_h3_comfy
  models/diffusion_models/...
  models/text_encoders/...
  models/vae/...
```

Network Volume is **region-scoped**. That limits GPU availability. RunPod **Model Caching** downloads a Hugging Face repo into a managed path and prefers hosts that already have the bytes. Cache is still under `/runpod-volume/...`, but it is **RunPod-managed**, not the operator’s hand-laid volume tree.

**Product path stays:** headless ComfyUI + four T2V weights + SaveVideo history + base64 or `BUCKET_*`.  
**This change only:** how the four weights appear under `/comfyui/models` at boot.

## Goal

1. Boot without a **user-attached** Network Volume that contains `minimax_h3_comfy/models/`, **when** Model Cache is configured with a **slim** (or proven-full) HF repo.  
2. Prefer RunPod HF cache at  
   `/runpod-volume/huggingface-cache/hub/models--{org}--{name}/snapshots/<hash>/`.  
3. Materialize Comfy layout under **`/models`** with **symlinks only** (no multi-GB copy onto container disk).  
4. Keep **legacy user volume** as the only offline fallback.  
5. Do **not** claim multi-region production readiness until **G0** (and slim-repo publish or full-repo proof) passes.

## Non-goals (v1)

- Changing ComfyUI pin, workflow inject map, or SaveVideo contract.  
- Runtime HF download of weights inside the worker (`ALLOW_HF_DOWNLOAD` / boot-time `download_weights` for inference).  
- Selective quantization filter on RunPod side (platform limitation — mitigated by **slim repo**).  
- Baking weights into the Docker image.  
- cu128 / Blackwell (5090, PRO 6000) — separate workstream.  
- Requiring S3 / RunPod S3 for delivery (base64 remains valid).  
- Multiple cached models per endpoint.  
- Relying on undocumented `org/name:hash` Model field syntax until live-proven.  
- Assuming full `Comfy-Org/MiniMax-H3` (~465 GB) is a viable default for Model Cache.

## Constraints & facts

| Fact | Implication |
|------|-------------|
| Cache layout = HF hub convention | Resolve snapshot under `HF_CACHE_ROOT`; not `MODEL_DIR/models` by default |
| Comfy needs `models/{diffusion_models,text_encoders,vae}/` | Adapter links 4 files into `/models/...` |
| HF paths for T2V four | `diffusion_models/…`, `text_encoders/…`, `vae/…` |
| Full `Comfy-Org/MiniMax-H3` ≈ **465 095 223 683 B** (~465 GB) | RunPod downloads **all** variants in the repo ([model caching limitations](https://docs.runpod.io/serverless/endpoints/model-caching)) |
| Four required files ≈ **42 470 585 471 B** (~42.47 GB) | Slim repo target size; worker never needs the rest |
| Container disk **45 GB** | Fits OS/image/outputs **only if weights are not written** to container disk. **Forbidden:** copy/download of weights to container disk in v1 |
| Cache path still `/runpod-volume/...` | Inspect `huggingface-cache` before assuming “no volume” |
| Model identity | Env **`MODEL_NAME`** = `org/name` matching endpoint **Model** field; default below |
| GPU allowlist (ops) | 48 GB+ Ada/Ampere/Hopper under cu124; no Blackwell until cu128 |
| RunPod Network Volume guidance | Large models often need ~500 GB+ volumes when not using slim cache |

## Pre-implementation gates (required)

### G0 — Cache size strategy (one of A or B must pass before production claim)

Worker code can land under G0 incomplete, but **“multi-region / no user volume” is not verified** until G0 passes.

#### Path A (recommended, primary design): slim HF repo

Publish (or use existing) a **public or token-gated** HF repo that contains **exactly** the four T2V weights with the same relative paths:

```text
diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
vae/minimax_h3_video_vae_fp16.safetensors
vae/minimax_h3_audio_vae_fp32.safetensors
```

| Check | Pass |
|-------|------|
| Repo total size | ≈ 42.5 GB (± small metadata), **not** 465 GB |
| Endpoint Model field | Slim `org/name` only (no `:hash` in v1) |
| Worker `MODEL_NAME` | Same string |
| Store fill | Completes on at least one datacenter; worker sees snapshot + 4 files |
| Multi-region | Document which regions/DCs actually received the cache (empirical log) |

Default production `MODEL_NAME` = **slim repo id** (fill after publish, e.g. `YOUR_ORG/MiniMax-H3-T2V-comfy-four` — exact id TBD in README/PINS).

#### Path B (exception): full `Comfy-Org/MiniMax-H3`

Only if product refuses a slim mirror:

| Check | Pass |
|-------|------|
| Live endpoint with Model = `Comfy-Org/MiniMax-H3` | Store accepts and finishes **~465 GB** download |
| Datacenter / region evidence | Cache present where GPU allowlist actually schedules |
| Worker | Still only **symlinks** the four files; ignores rest |
| Cost/time | Record wall time and store capacity impact |

**Fail closed:** if neither A nor B is done, deploys must keep **legacy Network Volume** (or accept that cache path is experimental).

### G1 — Model field format

- v1: endpoint Model = plain `org/name` (**`main` branch**).  
- Do **not** encode pins as `org/name:hash` in code or docs until RunPod’s live UI/API format is confirmed on a working endpoint.  
- After live proof, optional follow-up may document pin syntax; not part of v1 acceptance.

## Exact four weights (unchanged)

```text
diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
vae/minimax_h3_video_vae_fp16.safetensors
vae/minimax_h3_audio_vae_fp32.safetensors
```

Single source of truth: shared constant with `download_weights.WEIGHTS` / `PINS.md`  
(`download_weights.py` remains an **operator CLI** for filling a **user volume offline**, not a serverless boot path).

## Target boot architecture

```text
start.sh
  → python -u /app/model_store.py          # always materializes /models; exit ≠0 on fail
       priority:
         1) RunPod HF cache snapshot → symlink four files into /models
         2) Legacy MODEL_DIR/models → symlink (or link tree) into /models
  → verify four files under /models with du -hL
  → rm -rf "${COMFYUI_PATH}/models"        # required: stock Comfy ships a real directory
  → ln -sfn /models "${COMFYUI_PATH}/models"
  → ComfyUI → handler.py
```

```text
/runpod-volume/huggingface-cache/hub/
  models--{org}--{name}/
    refs/main
    snapshots/<hash>/
      diffusion_models/...
      text_encoders/...
      vae/...
      (other files only if full repo — ignore)

/models/                    # HARD-CODED materialize root (v1 — not configurable)
  diffusion_models/*.safetensors → symlink
  text_encoders/...
  vae/...

/comfyui/models → /models   # after rm -rf of previous models dir/link
```

If snapshot stores files under `models/diffusion_models/...`, resolver **must** accept both layouts.

## Python ↔ shell contract (normative)

| Side | Responsibility |
|------|----------------|
| `python /app/model_store.py` | Resolve source; **always** write Comfy tree at hard-coded **`/models`**; log `[ModelStore] ...` to stdout; exit **0** on success, **non-zero** on failure. Shell does **not** parse JSON or capture env exports. |
| `start.sh` | (1) Run `model_store.py`. (2) Check four paths under **`/models`** with **`du -hL`**. (3) **`rm -rf "${COMFYUI_PATH}/models"`** — Comfy image already has a normal `models/` directory; bare `ln -sfn` does **not** replace a directory. (4) **`ln -sfn /models "${COMFYUI_PATH}/models"`**. (5) Start Comfy/handler. |

Even for **legacy volume**, Python materializes into `/models` (symlinks to volume files) so shell logic stays single-path.

**Constant (code):** `COMFY_MODELS = Path("/models")` — not an environment variable in v1.

## Module contract: `model_store.py`

### `resolve_snapshot_path(model_id, cache_root) -> Path`

1. Require `model_id` in `org/name` form (reject empty; reject `:` pin syntax in v1 with clear error if passed).  
2. `model_root = cache_root / f"models--{org}--{name}"`.  
3. If `refs/main` exists: read hash; if `snapshots/<hash>` is a directory → return it.  
4. Else if `snapshots/` has **exactly one** subdirectory → return it.  
5. Else **fail**: message lists cache root, model root, and snapshot candidate names (no silent “first sorted” pick).

### `materialize_comfy_models(sources: dict[rel, Path], dest: Path = Path("/models")) -> None`

- **`dest` is always `/models` in production CLI** (parameter exists only for unit tests with tmp dirs).  
- No env override.  
- Create parent dirs; for each of four rel paths: `symlink` to absolute source.  
- Idempotent: replace existing symlink/file at dest.  
- **Never** copy weight bytes.  
- Assert each dest path `is_symlink()` after link (unit-tested).

### `prepare_models() -> None` (CLI main)

**Priority:**

| Order | Condition | Action |
|-------|-----------|--------|
| 1 | Snapshot resolvable **and** all 4 weights found | materialize symlinks → **`/models`**; log `source=cache` |
| 2 | Legacy volume has all 4 under `MODEL_DIR/models` (optional candidate scan kept for ops) | materialize symlinks from those files → **`/models`**; log `source=volume` |
| — | else | exit ≠0: mention endpoint **Model** / `MODEL_NAME`, cache path, or attach volume + `MODEL_DIR` |

**No** step for runtime HF download.

### Logging (required)

```text
[ModelStore] MODEL_NAME=...
[ModelStore] HF_CACHE_ROOT=...
[ModelStore] Using snapshot: ...   # if cache
[ModelStore] linked <rel> -> <realpath>
[ModelStore] source=cache|volume
```

On failure: list `HF_CACHE_ROOT` (if present), missing rels, snapshot candidates if ambiguous.

## Environment

| Variable | Default | Notes |
|----------|---------|--------|
| `MODEL_NAME` | Slim repo id (TBD after publish); interim dev default may be `Comfy-Org/MiniMax-H3` **only** for G0 path B experiments | Must match endpoint Model field (`org/name`). Same name as RunPod / vLLM examples. |
| `HF_CACHE_ROOT` | `/runpod-volume/huggingface-cache/hub` | Docs path |
| `MODEL_DIR` | unset | Legacy user volume root containing `models/` — **fallback only** |
| `HF_TOKEN` / `HUGGING_FACE_TOKEN` | unset | **Not** used at boot in v1; endpoint UI token is for RunPod store; CLI `download_weights` for offline volume fill may still use token |

**Hard-coded (not env):** materialize root = **`/models`**.

**Removed from v1:** `HF_MODEL_ID`, `HF_SNAPSHOT`, `ALLOW_HF_DOWNLOAD`, **`COMFY_MODELS_ROOT`**.

## `start.sh` changes (behavioral)

| Before | After |
|--------|-------|
| Require `MODEL_DIR/models` or die | `python -u /app/model_store.py` or die |
| Optional volume adopt + direct volume → Comfy | Always `/models` after `model_store.py` |
| Prefer volume models over image placeholders: `rm -rf` then `ln -sfn` | **Keep the same two-step:** `rm -rf "${COMFYUI_PATH}/models"` then `ln -sfn /models "${COMFYUI_PATH}/models"` |
| `du -h` on path | **`du -hL`** so sizes reflect symlink targets (V2.3) |

**Normative Comfy models link (must not drop `rm -rf`):**

```bash
# /comfyui/models is a real directory in the image; ln -sfn cannot replace a directory.
if [[ -e "${COMFYUI_PATH}/models" || -L "${COMFYUI_PATH}/models" ]]; then
  rm -rf "${COMFYUI_PATH}/models"
fi
ln -sfn /models "${COMFYUI_PATH}/models"
```

Comfy start + readiness probe + handler launch **unchanged**.

## Files to touch (implementation checklist)

| Path | Change |
|------|--------|
| `workers/minimax_h3_comfy/model_store.py` | **New** — resolve, materialize, CLI main; dest hard-coded `/models` |
| `workers/minimax_h3_comfy/start.sh` | Call `model_store.py`; verify `/models` with `du -hL`; **`rm -rf` + `ln -sfn /models`** |
| `workers/minimax_h3_comfy/Dockerfile` | COPY `model_store.py` into `/app/` |
| `workers/minimax_h3_comfy/Dockerfile.bootcheck` | **COPY `model_store.py`** (and any other modules `start.sh` invokes). Today it only copies `entrypoint.sh` + `start.sh` — after this change boot smoke image **must** include `model_store.py` or V1 fails at import. |
| `workers/minimax_h3_comfy/tests/test_model_store.py` | **New** — `unittest`, no network |
| `workers/minimax_h3_comfy/tools/local_boot_smoke.sh` | Fake HF cache fixture; builds **Dockerfile.bootcheck** |
| `workers/minimax_h3_comfy/README.md` | Cache deploy, G0, slim repo, env table |
| `workers/minimax_h3_comfy/PINS.md` | Slim vs full repo; four weights; cache path |
| `download_weights.py` | Shared WEIGHTS export if useful; **operator volume fill only** |

**Out of touch:** `handler.py` product/delivery, `workflow.py`, `workflows/t2va_api.json`.

---

## Verification plan (for review / QA)

### V0 — Unit (local, no GPU, no HF network)

Use existing suite style: **`unittest`** (not pytest-required).

```bash
cd workers/minimax_h3_comfy && python -m unittest discover -s tests -v
```

| ID | Check | Pass criteria |
|----|-------|----------------|
| V0.1 | `resolve_snapshot_path` with `refs/main` | Returns that snapshot dir |
| V0.2 | No refs, **exactly one** snapshot subdir | Returns it |
| V0.3 | No refs, **two+** snapshot subdirs | Fails; error lists candidates |
| V0.4 | Missing hub / empty snapshots | Clear error (Model / cache root) |
| V0.5 | `materialize` top-level `diffusion_models/` layout | All 4 dest are **symlinks** |
| V0.6 | `materialize` under `models/` layout | Same |
| V0.7 | Never copies | `Path.is_symlink()` true; no large byte writes |
| V0.8 | Missing one of four | Fail |
| V0.9 | `prepare` cache over empty volume | `source=cache`; files under materialize dest (tmpdir in tests / `/models` in prod) |
| V0.10 | `prepare` no cache, volume present | `source=volume`; dest filled via symlinks |
| V0.11 | `prepare` no cache/volume | Non-zero exit |
| V0.12 | Existing `test_download_weights`, `test_workflow`, `test_delivery` | Still green via unittest |
| V0.13 | No `COMFY_MODELS_ROOT` / env override path | Code and docs use hard-coded `/models` only in prod CLI |

### V1 — Local boot smoke (container, fake weights)

| ID | Check | Pass criteria |
|----|-------|----------------|
| V1.1 | Fake HF cache + `BOOT_CHECK=1` | `[ModelStore] source=cache`, four `ok weight` with `du -hL` sizes, Comfy ready, exit 0 |
| V1.2 | No cache, fake legacy volume | `source=volume`, boot OK |
| V1.3 | No cache, no volume | Non-zero; error mentions Model / cache / `MODEL_DIR` (not download) |
| V1.4 | Disk | Only tiny fake files; no multi-GB writes |

### V2 — RunPod endpoint (ops + container logs)

**Endpoint config (production path A):**

- Model: **slim** `org/name` (G0 path A)  
- `MODEL_NAME` env = same  
- HF token if gated  
- GPU: A40 / A6000 / L40 / L40S / 6000 Ada / H100  
- Container disk 45 GB (weights **not** on this disk)  
- No user Network Volume required when cache ready  
- Docker Command empty  

| ID | Check | Pass criteria |
|----|-------|----------------|
| V2.1 | Worker starts | ENTRYPOINT / start.sh visible |
| V2.2 | Cache present | `[ModelStore] Using snapshot:` under `huggingface-cache/hub` |
| V2.3 | Four weights sizes | `ok weight: …` with **`du -hL`** showing GB-scale targets (~20G / ~15G / ~4.9G / ~578M) |
| V2.4 | Symlink mode | No container free-space collapse; links under `/models` |
| V2.5 | Comfy ready | `ComfyUI ready after Ns` |
| V2.6 | Handler + fitness | `entering runpod.serverless.start`; no `_cuda_init_check` fail |
| V2.7 | Healthy worker | Endpoint healthy / job-capable |
| V2.8 | G0 evidence | Slim store size ~42 GB **or** full-repo proof recorded for path B |

**Failure signatures:**

| Log signature | Likely cause |
|---------------|--------------|
| Snapshot not found / cache root missing | Model field / store not ready / `MODEL_NAME` mismatch |
| Multiple snapshots, no refs/main | Ambiguous cache; fail as designed |
| Snapshot OK, `MISSING weight` | Partial/incomplete store, wrong `MODEL_NAME` / snapshot, or dual-layout miss (files not where resolver looks) — not “full repo lacks T2V paths” (official `Comfy-Org/MiniMax-H3` includes them) |
| `No space left` | Accidental copy/download or output bloat |
| `sm_120` / no kernel image | Blackwell on cu124 |
| Old `missing .../minimax_h3_comfy/models` only | Image not rebuilt |

### V3 — Functional smoke (GPU)

| ID | Check | Pass criteria |
|----|-------|----------------|
| V3.1 | Job | `864×480`, duration 2–5, fixed seed |
| V3.2 | Delivery | base64 or `video_url` per BUCKET matrix |
| V3.3 | Media | Decodable MP4 (+ audio if graph emits) |
| V3.4 | History | SaveVideo `images[0]` contract |

### V4 — Dual-mode (legacy volume + cache preference)

| ID | Required? | Check | Pass criteria |
|----|-----------|-------|----------------|
| **V4.1** | **Yes (DoD)** | Volume-only: empty/missing cache, `MODEL_DIR` points at four-file tree | `source=volume`; boot healthy (or V1/V2-equivalent logs); Goal legacy fallback proven |
| V4.2 | Optional | Cache + volume both present | Prefer cache when complete |

**V4.1 alone does not satisfy migration DoD** (need V2 cache path too).  
**V2 alone does not satisfy DoD either** — Goal requires keeping legacy volume fallback, so **V4.1 is mandatory in addition to V2.**

V4.1 may be proven via **V1.2** (local boot smoke, fake volume) **and/or** a volume-attached RunPod worker; evidence must show `source=volume` without relying on cache.

---

## Acceptance criteria (definition of done)

Migration is **done** only when **all** of the following hold:

| Layer | Required |
|-------|----------|
| Code | **V0** + **V1** pass (including **Dockerfile.bootcheck** with `model_store.py`) |
| Cache strategy | **G0 path A or B** documented with evidence |
| RunPod **cache path** | **V2.1–V2.7 all pass** on a worker that boots from Model Cache **without** a user-filled four-file `minimax_h3_comfy` volume |
| **Legacy volume fallback** | **V4.1 pass** (volume-only path still works) — **required**, not optional |
| Docs | README/PINS: slim repo id, `MODEL_NAME`, hard-coded `/models`, `rm -rf` before symlink, no runtime download, `du -hL`, G0 |
| Multi-region claim | Only after G0 + empirical schedule on >1 relevant DC/GPU set |
| V3 | Pass once **or** explicit pending with blocker (VRAM/job) — may lag V2 but does not replace V2 |
| V4.2 | Optional |

**Explicitly insufficient for DoD:**

- V0/V1 only  
- V4.1 volume-only green **without** V2 cache path  
- V2 cache green **without** V4.1 volume fallback  
- Code merged with G0 open and no live cache worker  

**Not required for DoD:** S3 delivery, cu128, full-repo as default, `org/name:hash` pins, multi-region beyond single-DC V2 proof (multi-region is a **separate claim** after G0+evidence), V4.2.

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Full repo 465 GB unusable as default | High | **Slim repo is primary design**; G0 path B is exception with proof |
| 45 GB disk + any weight download | High | **No runtime download in v1**; volume or cache only |
| Wrong model identity / cache miss | High | Single `MODEL_NAME` = endpoint Model; main only |
| Ambiguous multi-snapshot | Med | Fail with candidate list |
| Shell/Python desync | Med | Hard-coded `/models` only; no env split |
| `ln -sfn` onto existing directory | High | Mandatory `rm -rf` before link (document + keep in start.sh) |
| Bootcheck image missing `model_store.py` | Med | Update **Dockerfile.bootcheck** in same PR as start.sh |
| `du` without `-L` understates size | Low | Mandate `du -hL` in start.sh |
| Claiming done via volume-only | High | DoD requires **V2 cache path** |
| Claiming done via cache-only (legacy broken) | High | DoD also requires **V4.1** volume fallback |

## Out of scope follow-ups

- Runtime download after larger container disk (if ever).  
- Configurable materialize root (`COMFY_MODELS_ROOT`) — rejected for v1.  
- Snapshot pin env after RunPod format is proven live.  
- Torch cu128 multi-arch image.  
- Non-RunPod R2/S3 output.  
- Private Model Repository (RunPod beta) if it supersedes HF slim mirror.

## Implementation order

1. **G0 decision:** publish slim repo (A) **or** schedule full-repo store experiment (B).  
2. `model_store.py` + unittest (V0).  
3. `start.sh` (`du -hL`, **`rm -rf` + `ln -sfn /models`**).  
4. **Dockerfile** + **Dockerfile.bootcheck** both COPY `model_store.py`.  
5. `local_boot_smoke` fake cache (V1).  
6. README/PINS (`MODEL_NAME`, G0, fixed `/models`).  
7. Rebuild → **V2 (required)** + **V4.1 (required)** → V3; V4.2 optional.

---

## Verification sign-off (fill after runs)

| Gate | Date | Result | Evidence |
|------|------|--------|----------|
| G0 slim repo (A) or full proof (B) | | ☐ pass / ☐ fail / ☐ n/a | |
| V0 unit (`python -m unittest`) | 2026-08-10 | ☑ pass | `python -m unittest discover -s tests -v` — 40 OK |
| V1 boot smoke (**Dockerfile.bootcheck** + `model_store.py`) | 2026-08-10 | ☑ pass | `tools/local_boot_smoke.sh` — missing/cache/volume/incomplete |
| **V2 RunPod healthy (cache path) — required for DoD** | | ☐ pass / ☐ fail | |
| **V4.1 volume-only fallback — required for DoD** | 2026-08-10 | ☑ local (V1.2) / ☐ RunPod | boot smoke `source=volume`; live endpoint still optional evidence |
| V3 first T2V job | | ☐ pass / ☐ fail / ☐ pending | |
| V4.2 cache+volume preference (optional) | 2026-08-10 | ☑ unit | `test_cache_preferred_over_volume` |
| Docs updated | 2026-08-10 | ☑ pass | README + PINS |
| Multi-region claim allowed | | ☐ yes / ☐ no | needs G0 + V2 |

**Implementer:** agent (local V0/V1)  
**Reviewer:**  
