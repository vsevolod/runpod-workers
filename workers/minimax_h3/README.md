# RunPod worker — MiniMax H3 T2V

Thin serverless worker for **text-to-video+audio (t2va)** via Hugging Face
**diffusers ModularPipeline**. No ComfyUI.

| | |
|--|--|
| **Model** | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (t2va half) |
| **Backend** | diffusers ModularPipeline pin `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc` (PR #14355) |
| **GPU** | **1× 80 GB** class (A100/H100) + auto CPU offload; not a 24 GB Comfy quant path |
| **Steps** | Fixed **50** (product parity; not exposed in API) |
| **FPS** | 24; frames snapped to `17n+5` |

## License (read before deploy)

MiniMax H3 is under the **MiniMax H3 Community License Agreement**. Full text:
[`LICENSES/MINIMAX-H3-COMMUNITY.txt`](LICENSES/MINIMAX-H3-COMMUNITY.txt).

- **Excluded Territories:** EU, UK, Republic of Korea, United States.
- The Agreement applies to **deployers (Hosted Services)** and requires binding
  **downstream users** to restrictions at least as protective as Section V and
  Exhibit A (Acceptable Use Policy).
- Do not deploy or serve users in Excluded Territories under this Community
  License without a separate MiniMax authorization.
- Redistribution notice (License § III.4) is also in [`NOTICE`](NOTICE).

This repository does **not** implement product-level geo-blocking; that is an
ops/product obligation.

## Layout

```
workers/minimax_h3/
├── Dockerfile
├── handler.py
├── schemas.py
├── download_weights.py
├── requirements.txt
├── test_input.json
├── h3_infer/
│   ├── duration.py
│   ├── canvas.py
│   ├── request.py
│   ├── delivery.py
│   └── pipeline.py
└── tests/
```

## Network volume

| Component | ~size |
|-----------|------:|
| t2va half (`transformer/`, `text_encoder/`, VAEs, …) | **~144 GB** |
| Recommended volume | **≥ 200 GB** |
| Host RAM | large (bf16 TE+DiT with offload) |

```text
/runpod-volume/minimax_h3/
  modular_model_index.json
  transformer/
  text_encoder/
  vae/
  audio_vae/
  tokenizer/
  processor/
  scheduler/
  audio_scheduler/
```

Bootstrap on a Pod with the volume attached (not inside a serverless job):

```bash
# image has deps; or pip install huggingface-hub
export HF_TOKEN=hf_...   # if needed
export HF_XET_HIGH_PERFORMANCE=1   # optional
python download_weights.py --output /runpod-volume/minimax_h3

# dry-run (no network):
python download_weights.py --output /tmp/minimax_h3_dry --dry-run
```

Ref2VA / `transformer_ref/` are **not** downloaded (T2V MVP only).

## Endpoint env

| Variable | Required | Notes |
|----------|----------|--------|
| `MODEL_DIR` | yes | default `/runpod-volume/minimax_h3` |
| `LOCAL_FILES_ONLY` | recommended after bootstrap | `1` / `true` |
| `BUCKET_ENDPOINT_URL` | for URL delivery | all four required |
| `BUCKET_ACCESS_KEY_ID` | for URL delivery | |
| `BUCKET_SECRET_ACCESS_KEY` | for URL delivery | |
| `BUCKET_NAME` | for URL delivery | passed to `upload_file_to_bucket` |
| `REQUIRE_BUCKET` | optional | `1` → process exits if any of the four missing |
| `MAX_INLINE_VIDEO_BYTES` | optional | default `7000000` when no bucket |
| `LOG_LEVEL` | optional | default `INFO` |

Without a complete bucket config, small MP4s return `video` (raw base64); large
files return an error (no fake `local_upload/` URL).

## API

```json
{
  "input": {
    "prompt": "string (required)",
    "width": 864,
    "height": 480,
    "duration": 5.0,
    "seed": -1
  }
}
```

| Field | Default | Constraints |
|-------|---------|-------------|
| `prompt` | — | non-empty, ≤ 8000 chars |
| `width` / `height` | 864×480 | multiples of 32; aspect ∈ [1/4, 4]; area ≤ packing nominal for aspect |
| `duration` | 5.0 | **[5.0, 14.375]** seconds → frames `17n+5` @ 24 fps |
| `seed` | -1 | `-1` → random; else ≥ 0 |

**Canvas:** packing `resolve_canvas_size` parity (not “max long edge 1344”).
Official long canvas **2016×512** is valid; preview **864×480** is valid.

**Success (URL):**

```json
{
  "video_url": "https://…",
  "width": 864,
  "height": 480,
  "length": 124,
  "seed": 42,
  "requested_duration": 5.0,
  "output_duration": 5.1666…,
  "fps": 24,
  "model": "MiniMaxAI/MiniMax-H3"
}
```

**Success (inline):** field `video` (raw base64 MP4, no `data:` prefix) when no
bucket and size ≤ inline limit — same metadata fields as URL mode.

**Errors:** `{"error": "…"}`; CUDA OOM and unexpected failures set
`"refresh_worker": true`.

## Deploy

- Dockerfile path: `workers/minimax_h3/Dockerfile`
- Build context: **repository root**
- GPU: A100 80GB / H100 class recommended for MVP bf16 + offload
- FlashBoot: models load once at process start (`python -u handler.py`)

## Tests (CPU)

```bash
cd workers/minimax_h3 && PYTHONPATH=. python3.12 -m unittest discover -s tests -p 'test_*.py' -v
```

GPU smoke (allowed territory only): bootstrap weights, deploy, run
`test_input.json` → expect `length=124` and `video_url` https with audio muxed.

## Level-1 spike (R&D — not production default)

Thin-Python path for Comfy-Org **non-pruned** FL2VA int8 ConvRot DiT into stock
diffusers `MiniMaxH3Transformer3DModel` (no ComfyUI), official TE/VAE. See:

- Spec: `docs/superpowers/specs/2026-08-04-minimax-h3-level1-comfy-dit-spike-design.md`
- Plan: `docs/superpowers/plans/2026-08-04-minimax-h3-level1-comfy-dit-spike.md`
- Report: `docs/superpowers/specs/2026-08-04-minimax-h3-level1-comfy-dit-spike-report.md`

```bash
# Hybrid weights (~112 GB volume; not the 144 GB official DiT shards)
python download_weights.py --output /runpod-volume/minimax_h3_hybrid \
  --pack hybrid_spike --also-fetch-pruned-for-g0

# G0 inspect
PYTHONPATH=. python tools/spike_inspect_comfy_dit.py \
  /runpod-volume/minimax_h3_hybrid/comfy/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors

# G4 fail-closed forward (needs GPU host + local transformer/config.json)
PYTHONPATH=. python tools/spike_dit_forward.py \
  --dit …/minimax_h3_fl2va_int8_convrot.safetensors \
  --model-dir /runpod-volume/minimax_h3_hybrid
```

**Production `H3Pipeline` / handler still uses official MiniMaxAI pack only.**

## Status

MVP implementation. **Not production-ready** until GPU smoke on a real 80 GB
pod succeeds in an allowed territory.
