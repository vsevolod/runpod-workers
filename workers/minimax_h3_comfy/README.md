# MiniMax H3 — ComfyUI serverless worker

Headless **native** ComfyUI MiniMax H3 T2V (pruned int8 DiT + Qwen3-VL TE + video/audio VAEs). No required custom-node pack. Output path from **SaveVideo** history only.

Design/plan:

- `docs/superpowers/specs/2026-08-05-minimax-h3-comfyui-serverless-design.md`
- `docs/superpowers/plans/2026-08-05-minimax-h3-comfyui-serverless.md`
- Model Cache migration: `docs/superpowers/specs/2026-08-10-minimax-h3-model-cache-design.md`
- Pins & inject map: [`PINS.md`](PINS.md)

## Product input

**T2V** (default — no images):

```json
{
  "input": {
    "prompt": "…",
    "width": 864,
    "height": 480,
    "duration": 5.0,
    "seed": 42
  }
}
```

**I2V / FL2V** — same four `fl2va` weights; optional keyframes:

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
| `first_image` | Optional. HTTPS URL, raw base64, or `data:image/{png,jpeg,jpg,webp};base64,…` → `first_frame` |
| `last_image` | Optional; requires `first_image` → `last_frame` |
| Mode | Implicit: no `first_image` → T2V; first only → I2V; both → FL2V. Response includes `mode` |

- Canvas: multiples of 32; short edge ≤ 768; long ≤ 1344 (client always sets width/height; defaults 864×480).
- Default canvas **864×480** (template 0.4 MP) until smoke metrics justify 1344×768.
- `seed: -1` → random.
- No raw workflow field; no explicit `mode` field.
- Image limits: 20 MiB, 4096² pixels (see `image_input.py`).

## Delivery

| `BUCKET_*` | Behavior |
|------------|----------|
| all four set | upload MP4 → `delivery: "s3"` + `bucket` + `key` (no URL) |
| all empty | `video` base64 if size ≤ `MAX_INLINE_VIDEO_BYTES` (default 7e6) |
| partial | **worker exits at start** |

S3 delivery is for files that exceed the RunPod response cap (`/run` 10 MB, `/runsync` 20 MB). The worker does **not** attach a Network Volume and does **not** return a presigned URL (RunPod S3 API has no presign). The consumer `GetObject`s with its own credentials.

| Env | Required for S3 | Meaning |
|-----|-----------------|--------|
| `BUCKET_ENDPOINT_URL` | yes | `https://s3api-<DC>.runpod.io/` |
| `BUCKET_ACCESS_KEY_ID` / `BUCKET_SECRET_ACCESS_KEY` | yes | RunPod **S3 API key** (Settings), not the account API key |
| `BUCKET_NAME` | yes | network volume id |
| `BUCKET_REGION` | no | if unset, parsed from `s3api-<dc>.runpod.io` → `<DC>` |

## Model sources (boot)

At boot, `model_store.py` materializes **symlinks only** under hard-coded **`/models`**, then `start.sh` does `rm -rf` + `ln -sfn /models → /comfyui/models`.

| Priority | Source | When |
|----------|--------|------|
| 1 | RunPod **Model Cache** (HF hub layout) | Endpoint **Model** field = `MODEL_NAME` (`org/name`); store filled under `/runpod-volume/huggingface-cache/hub/` |
| 2 | Legacy **Network Volume** | `MODEL_DIR` root contains `models/` with the four T2V files |

**No runtime HF download** in the worker (45 GB container disk cannot hold ~42 GB weights + image + outputs).

### Preferred: Model Cache (slim repo)

1. Publish or use a **slim** HF repo with **exactly** the four T2V paths (~42.5 GB), **not** the full `Comfy-Org/MiniMax-H3` (~465 GB) unless G0 path B is proven.
2. Endpoint → **Model** = slim `org/name` (main only; no `:hash` pin in v1).
3. Env `MODEL_NAME` = same string.
4. Optional HF token in RunPod UI if the repo is gated.
5. No user Network Volume required once the store is ready.

Default interim `MODEL_NAME` / image env: `Comfy-Org/MiniMax-H3` (for G0 path B experiments only until a slim id is published — see `PINS.md`).

### Legacy volume bootstrap (operator offline)

```bash
python download_weights.py --output /runpod-volume/minimax_h3_comfy --dry-run
python download_weights.py --output /runpod-volume/minimax_h3_comfy
```

Exactly four files under `{output}/models/…` (see `PINS.md`). Set `MODEL_DIR` to that root when not using Model Cache.

## Comfy pin

Image clones ComfyUI at **v0.30.0** (`b1693ec…`). worker-comfyui tags alone are not enough if older than H3.

## Headless smoke (GPU host)

```bash
# ComfyUI running with pin + four weights
python tools/headless_t2v_smoke.py \
  --comfy-url http://127.0.0.1:8188 \
  --output-dir /path/to/ComfyUI/output \
  --width 864 --height 480 --duration 2
```

Must resolve MP4 via SaveVideo `outputs["92"]["images"][0]`, then `ffprobe` audio.

## Tests (CPU)

```bash
cd workers/minimax_h3_comfy
python -m unittest discover -s tests -v
```

## Local boot smoke (no RunPod, no CUDA, fake weights)

Validates entrypoint + `start.sh` + `model_store.py` (cache **and** volume paths) + mock Comfy `/system_stats`.
Uses slim `Dockerfile.bootcheck` (must COPY `model_store.py`).

```bash
# from monorepo root
./workers/minimax_h3_comfy/tools/local_boot_smoke.sh
```

Cases: missing source (fail), fake HF cache (`source=cache`), fake volume (`source=volume`), incomplete weights (fail).

Manual cache fixture:

```bash
./workers/minimax_h3_comfy/tools/make_fake_hf_cache.sh /tmp/mh3_hub
docker build -f workers/minimax_h3_comfy/Dockerfile.bootcheck \
  -t minimax-h3-bootcheck workers/minimax_h3_comfy
docker run --rm \
  -v /tmp/mh3_hub:/runpod-volume/huggingface-cache/hub \
  -e MODEL_NAME=Comfy-Org/MiniMax-H3 \
  -e HF_CACHE_ROOT=/runpod-volume/huggingface-cache/hub \
  -e BOOT_CHECK=1 -e BOOT_FAIL_SLEEP=0 \
  minimax-h3-bootcheck
# expect: source=cache … ok weight … ComfyUI ready … BOOT_CHECK ok
```

Full CUDA image still requires a real GPU + real weights; use bootcheck for daily iteration.

## Deploy (RunPod)

- Dockerfile path: `workers/minimax_h3_comfy/Dockerfile`
- Build context: repo root
- **Docker Command** empty (use image `CMD`)
- **Model Cache (preferred):**
  - Endpoint **Model** = slim `org/name` (G0 path A) or proven full repo (path B)
  - Env `MODEL_NAME` = same; `HF_CACHE_ROOT` default `/runpod-volume/huggingface-cache/hub`
  - Container disk **45 GB** (weights stay on cache volume via symlinks — never copy)
  - No user Network Volume required when cache is ready
- **Legacy volume (fallback):**
  - Attach Network Volume + `MODEL_DIR` = root containing `models/`
  - Fill offline with `download_weights.py`
- Env: all four `BUCKET_*` **or** none (optional `BUCKET_REGION`)
- GPU allowlist (cu124): A40 / A6000 / L40 / L40S / 6000 Ada / H100 — **not** Blackwell until cu128
- Image includes `build-essential` so Triton can JIT-compile CUDA utils at runtime

### "All workers are unhealthy" / empty Container logs

RunPod **System** logs only show `start/remove container`. Real output is **Container** logs from CMD.

If Container is empty and worker `exit code 1`:

1. **Endpoint → Docker configuration → Docker Command / Start command**  
   Must be **empty** (use image `CMD`).  
   If you set a custom command, clear it or set exactly:  
   `/bin/bash /app/entrypoint.sh`
2. Rebuild after `entrypoint.sh` (prints immediately, `bash -x`, sleeps 120s on failure so logs stay visible).
3. **Cache path:** set Model field + `MODEL_NAME`; wait for store; look for `[ModelStore] source=cache` and `Using snapshot:`.
   **Volume path:** attach volume + `MODEL_DIR` containing `models/`.
4. `BUCKET_*`: all four or none.

Typical app log lines (after rebuild):

| Log | Fix |
|-----|-----|
| `ENTRYPOINT ...` then `No usable model source` / `model_store.py failed` | Model field / `MODEL_NAME` / cache not ready, or volume + `MODEL_DIR` |
| Snapshot / cache root missing | Store not filled or `MODEL_NAME` mismatch |
| `MISSING weight` | Incomplete store/volume; re-fill slim repo or `download_weights.py` |
| `Partial BUCKET_*` | Fix env |
| `sm_120` / no kernel image | Blackwell GPU on cu124 image — use allowlist GPUs |
| `ComfyUI process died` | Scroll Comfy log tail in same log stream |

## Metrics (fill after Phase 1 / 4 smoke)

| Metric | Value |
|--------|--------|
| Peak VRAM | _pending_ |
| Cold start | _pending_ |
| Warm 5s job | _pending_ |
| Default canvas | 864×480 (provisional) |
