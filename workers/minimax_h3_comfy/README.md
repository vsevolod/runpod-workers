# MiniMax H3 — ComfyUI serverless worker

Headless **native** ComfyUI MiniMax H3 T2V (pruned int8 DiT + Qwen3-VL TE + video/audio VAEs). No required custom-node pack. Output path from **SaveVideo** history only.

Design/plan:

- `docs/superpowers/specs/2026-08-05-minimax-h3-comfyui-serverless-design.md`
- `docs/superpowers/plans/2026-08-05-minimax-h3-comfyui-serverless.md`
- Pins & inject map: [`PINS.md`](PINS.md)

## Product input

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

- Canvas: multiples of 32; short edge ≤ 768; long ≤ 1344.
- Default canvas **864×480** (template 0.4 MP) until smoke metrics justify 1344×768.
- `seed: -1` → random.
- No raw workflow field in v1.

## Delivery

| `BUCKET_*` | Behavior |
|------------|----------|
| all four set | `video_url` via S3-compatible upload |
| all empty | `video` base64 if size ≤ `MAX_INLINE_VIDEO_BYTES` (default 7e6) |
| partial | **worker exits at start** |

## Volume bootstrap

```bash
python download_weights.py --output /runpod-volume/minimax_h3_comfy --dry-run
python download_weights.py --output /runpod-volume/minimax_h3_comfy
```

Exactly four files under `{output}/models/…` (see `PINS.md`).

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

Validates entrypoint + `start.sh` + `MODEL_DIR`/four weight paths + mock Comfy `/system_stats`.
Uses slim `Dockerfile.bootcheck` (~python slim, seconds to rebuild).

```bash
# from monorepo root
./workers/minimax_h3_comfy/tools/local_boot_smoke.sh
```

Only fake weight tree:

```bash
./workers/minimax_h3_comfy/tools/make_fake_weights.sh /tmp/mh3_fake
docker build -f workers/minimax_h3_comfy/Dockerfile.bootcheck \
  -t minimax-h3-bootcheck workers/minimax_h3_comfy
docker run --rm \
  -v /tmp/mh3_fake:/runpod-volume/minimax_h3_comfy \
  -e MODEL_DIR=/runpod-volume/minimax_h3_comfy \
  -e BOOT_CHECK=1 -e BOOT_FAIL_SLEEP=0 \
  minimax-h3-bootcheck
# expect: ENTRYPOINT … ok weight … ComfyUI ready … BOOT_CHECK ok
```

Full CUDA image still requires a real GPU + real weights; use bootcheck for daily iteration.

## Deploy (RunPod)

- Dockerfile path: `workers/minimax_h3_comfy/Dockerfile`
- Build context: repo root
- **Network Volume** attached to the endpoint (same datacenter as download)
- Env:
  - `MODEL_DIR` = volume root that contains `models/`  
    (default `/runpod-volume/minimax_h3_comfy`; if you downloaded to `/workspace/minimax_h3_comfy`, set that)
  - all four `BUCKET_*` **or** none
- GPU: measure after first smoke; start conservatively (e.g. 48 GB) until peak known
- Image includes `build-essential` so Triton can JIT-compile CUDA utils at runtime

### "All workers are unhealthy" / empty Container logs

RunPod **System** logs only show `start/remove container`. Real output is **Container** logs from CMD.

If Container is empty and worker `exit code 1`:

1. **Endpoint → Docker configuration → Docker Command / Start command**  
   Must be **empty** (use image `CMD`).  
   If you set a custom command, clear it or set exactly:  
   `/bin/bash /app/entrypoint.sh`
2. Rebuild after `entrypoint.sh` (prints immediately, `bash -x`, sleeps 120s on failure so logs stay visible).
3. Attach Network Volume + set `MODEL_DIR` to the folder that contains `models/`.
4. `BUCKET_*`: all four or none.

Typical app log lines (after rebuild):

| Log | Fix |
|-----|-----|
| `ENTRYPOINT ...` then `missing .../models` | Volume / `MODEL_DIR` |
| `MISSING weight` | Re-run `download_weights.py` |
| `Partial BUCKET_*` | Fix env |
| `ComfyUI process died` | Scroll Comfy log tail in same log stream |

## Metrics (fill after Phase 1 / 4 smoke)

| Metric | Value |
|--------|--------|
| Peak VRAM | _pending_ |
| Cold start | _pending_ |
| Warm 5s job | _pending_ |
| Default canvas | 864×480 (provisional) |
