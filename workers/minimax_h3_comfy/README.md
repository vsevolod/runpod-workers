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

## Deploy (RunPod)

- Dockerfile path: `workers/minimax_h3_comfy/Dockerfile`
- Build context: repo root
- Env: `MODEL_DIR=/runpod-volume/minimax_h3_comfy`, optional full `BUCKET_*`
- GPU: measure after first smoke; start conservatively (e.g. 48 GB) until peak known
- Image includes `build-essential` so Triton can JIT-compile CUDA utils at runtime

## Metrics (fill after Phase 1 / 4 smoke)

| Metric | Value |
|--------|--------|
| Peak VRAM | _pending_ |
| Cold start | _pending_ |
| Warm 5s job | _pending_ |
| Default canvas | 864×480 (provisional) |
