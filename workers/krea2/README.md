# RunPod worker — Krea 2 Turbo (FP8 / INT8 ConvRot)

Thin serverless worker: text-to-image **and** identity image edit via dual
conditioning. No ComfyUI.

| | |
|--|--|
| **Model (default)** | [AlperKTS/Krea2_FP8](https://huggingface.co/AlperKTS/Krea2_FP8) DiT + official [krea-ai/krea-2](https://github.com/krea-ai/krea-2) sampler |
| **Model (optional)** | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) `krea2_turbo_int8_convrot.safetensors` via `DIT_QUANT=int8_convrot` |
| **Text encoder** | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (HF, bf16) |
| **VAE** | Clean Diffusers VAE from [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) (`vae/`); local Comfy single-file is tried first and **ignored** if keys do not match |
| **GPU** | **24 GB** class (TE offload after encode; all-resident OOMs at VAE decode). INT8 path is preferred on **RTX 30xx (Ampere)** |
| **Template patterns** | [worker-sdxl](https://github.com/runpod-workers/worker-sdxl) |

## Layout

```
workers/krea2/
├── Dockerfile              # Dockerfile path for RunPod; see build context below
├── handler.py              # RunPod entry + ModelHandler load-once
├── schemas.py
├── download_weights.py     # volume bootstrap (not baked into image)
├── test_input.json
├── requirements.txt
└── krea2_infer/            # vendored krea-2 + FP8 / INT8 loaders
```

## Network volume

Mount the same datacenter volume on the endpoint.

### Disk budget (this worker, not Comfy)

| Component | Source | ~size |
|-----------|--------|------:|
| DiT Turbo FP8 | `krea2_turbo_fp8.safetensors` (AlperKTS) | ~12 GB |
| DiT Turbo INT8 ConvRot (optional) | `krea2_turbo_int8_convrot.safetensors` (Comfy-Org) | ~13.5 GB |
| Text encoder | HF `Qwen/Qwen3-VL-4B-Instruct` (bf16 snapshot) | ~8–10 GB |
| VAE | HF `Qwen/Qwen-Image` `vae/` (preferred); optional local single-file | ~0.3 GB |
| **Total (FP8 path)** | | **~20–23 GB** |
| Recommended volume | models + HF cache slack (+ LoRA / second DiT) | **≥25–30 GB** |

`≥40 GB` is only “plenty of headroom”, not a hard requirement.

```text
/runpod-volume/krea2/
  krea2_turbo_fp8.safetensors              # default path (~12 GB)
  krea2_turbo_int8_convrot.safetensors     # optional; needs DIT_QUANT=int8_convrot
  qwen_image_vae.safetensors               # optional; ignored if not Diffusers-compatible
  loras/                                   # optional runtime LoRA adapters
    lora_a.safetensors
    lora_b.safetensors
/runpod-volume/hf/                         # HF_HOME: TE + clean VAE cache
```

Place Krea 2 adapter `.safetensors` files in `loras/` (or `$LORA_DIR`): standard
rank LoRA and/or full `weight_diff` files (e.g. Fedor-style
`diffusion_model.txtfusion.projector.diff`). The worker scans **only top-level**
`*.safetensors` stems at process start into an allowlist; files are loaded only
when selected for a job. Request `type` must match each file’s tensor layout.
After adding or removing adapters, restart warm workers so they re-scan.

Bootstrap on a Pod with the volume attached:

```bash
pip install huggingface-hub hf_transfer
export HF_TOKEN=hf_...   # if needed
export HF_HOME=/runpod-volume/hf
# Default FP8 DiT:
python download_weights.py --output /runpod-volume/krea2 --hf-home /runpod-volume/hf
# Or INT8 ConvRot (3090-friendly) / both:
python download_weights.py --output /runpod-volume/krea2 --hf-home /runpod-volume/hf --quant int8_convrot
python download_weights.py --output /runpod-volume/krea2 --hf-home /runpod-volume/hf --quant both
```

`download_weights.py` also snapshots `Qwen/Qwen-Image` `vae/*` into `HF_HOME` (clean Diffusers VAE). Comfy-format `qwen_image_vae.safetensors` from AlperKTS is optional and often **not** loadable via Diffusers `from_single_file`.

### Comfy FP8 TE is not used

| File | Used here? |
|------|------------|
| `krea2_turbo_fp8.safetensors` | **Yes** (DiT, default `DIT_QUANT=fp8`) |
| `krea2_turbo_int8_convrot.safetensors` | **Yes** when `DIT_QUANT=int8_convrot` |
| `qwen_image_vae.safetensors` | Optional try; else clean HF VAE |
| `qwen3vl_4b_fp8_scaled.safetensors` (~5.2 GB) | **No** — ComfyUI `text_encoders/` format |

Text encoding follows **official krea-2**: `Qwen3VLForConditionalGeneration` from Hugging Face. That is larger on disk/VRAM than Comfy’s FP8 TE, but needs no Comfy graph or scaled-FP8 TE loader. Cache the HF model on the volume so cold starts do not re-download it.

### DiT quant modes (`DIT_QUANT`)

| `DIT_QUANT` | DiT file (auto under `MODEL_DIR`) | Compute path |
|-------------|-----------------------------------|--------------|
| `fp8` (default / unset) | `krea2_turbo_fp8.safetensors` | FP8 storage, cast to bf16 on Linear forward (current production) |
| `int8_convrot` (aliases: `int8`, `int8_tensorwise`) | `krea2_turbo_int8_convrot.safetensors` | int8_tensorwise + online ConvRot; CUDA `torch._int_mm` W8A8 when batch is large enough, else dequant fallback |

Mode is fixed at worker start (FlashBoot-friendly). Mismatched file vs mode fails fast. TE / VAE / API are shared.

**Why INT8 on 3090:** Ampere has INT8 Tensor Cores but not FP8. Native INT8 GEMM avoids the FP8→BF16 cast tax. Speedup only applies when the INT8 path is active — plain dequant of an int8 file under `DIT_QUANT=fp8` is rejected.

## Endpoint env

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_DIR` | `/runpod-volume/krea2` | DiT / optional VAE files |
| `LORA_DIR` | `/runpod-volume/krea2/loras` | Directory of pre-placed LoRA `.safetensors` (scanned once at start) |
| `DIT_QUANT` | `fp8` | `fp8` or `int8_convrot` — selects DiT load + Linear runtime |
| `DIT_PATH` | auto under `MODEL_DIR` | Override DiT safetensors path (must match `DIT_QUANT`) |
| `VAE_PATH` | auto under `MODEL_DIR` | Override VAE safetensors path |
| `TEXT_ENCODER_ID` | `Qwen/Qwen3-VL-4B-Instruct` | HF id or local snapshot path |
| `HF_HOME` | (hf default) | Put on volume for faster/offline loads |
| `LOCAL_FILES_ONLY` | unset | `1` after cache is warm |
| `BUCKET_ENDPOINT_URL` | unset | If set, upload via RunPod S3 helpers instead of base64 |
| `TORCH_COMPILE_DISABLE` | `1` (Dockerfile) | Skip torch.compile / Inductor (no gcc in image) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` (Dockerfile) | Reduce CUDA allocator fragmentation |

## Deploy (RunPod)

Repo may be **public or private** (connect GitHub in RunPod for private). Do not commit tokens or weights.

1. Create Network Volume (**≥25–30 GB** free: FP8 DiT + HF TE + VAE; more if LoRAs).
2. Run `download_weights.py` on a **Pod** (not serverless) with that volume attached.
3. **Serverless → New Endpoint → Import Git Repository** (not an always-on Pod for serving)

### GitHub / Docker settings (monorepo)

| Setting | Value | Notes |
|---------|-------|--------|
| **Branch** | `master` (this repo) | Not `main` unless you rename |
| **Dockerfile path** | `workers/krea2/Dockerfile` | Path to the file only |
| **Build context** | `.` | **Repo root** — required because Dockerfile does `COPY workers/krea2/...` |

Wrong build context (e.g. `workers/krea2`) makes `COPY workers/krea2/...` fail during image build. Context does **not** control *when* a build starts; it only controls how Docker resolves paths.

Other endpoint knobs:

- GPU: **24 GB** class (A5000 / 3090 / 4090 / L4, etc.)
- Attach the volume (same DC as the endpoint)
- Container disk: 20+ GB
- FlashBoot: on
- Active workers: `0` (or `1` for warm)
- Env: `MODEL_DIR=/runpod-volume/krea2`, ideally `HF_HOME=/runpod-volume/hf`

### Updating the endpoint after code changes

A plain `git push` may **not** rebuild the worker. Per [RunPod GitHub integration](https://docs.runpod.io/serverless/workers/github-integration), create a **GitHub Release** (tag) to trigger a new build, then check the endpoint **Builds** tab.

```bash
git push origin master
gh release create krea2-vX.Y.Z --target master \
  --title "krea2: short summary" \
  --notes "What changed"
```

If a release appears on GitHub but **no** new row shows under Builds, re-check GitHub app access to this repo, or **Clone Endpoint** / re-import from GitHub (forces a fresh build from branch tip).

## API

### `type` field

| `type` | Meaning | `images` |
|--------|---------|----------|
| `image_generate` (default) | Text-to-image | must be empty / omitted |
| `image_edit` | Identity edit (source image + instruction) | **exactly one** base64 / data-URL entry |

`images` is always an **array** (future multi-ref). Edit currently requires length **1**.

### Text-to-image (`image_generate`)

```json
{
  "input": {
    "type": "image_generate",
    "prompt": "a fox walking in the snow",
    "width": 1024,
    "height": 1024,
    "seed": 42,
    "num_inference_steps": 8,
    "guidance_scale": 0.0,
    "mu": 1.15,
    "loras": [
      {"name": "lora_a", "strength": 0.8},
      {"name": "lora_b"},
      {"name": "fedor_bypass", "type": "weight_diff", "strength": 4.0}
    ]
  }
}
```

Turbo defaults: **8 steps**, **CFG 0**, **mu 1.15**. Width/height multiples of **16** (1024–2048).

### Identity edit (`image_edit`)

Dual conditioning (not a plain LoRA attach):

1. **Grounded Qwen3-VL encode** — instruction + source image through the vision TE (`mm_processor`).
2. **Source VAE tokens** — fit/crop pixels → encode → RoPE frame=1 concatenated with target noise (frame=0).

Primary reference implementation: [conradlocke/krea2-identity-edit](https://huggingface.co/spaces/conradlocke/krea2-identity-edit) **v1.2** Space (and ComfyUI-Krea2Edit). Place the identity LoRA on the volume:

```text
/runpod-volume/krea2/loras/krea2_identity_edit_v1_2.safetensors
```

Restart warm workers after adding the file so the catalog re-scans.

```json
{
  "input": {
    "type": "image_edit",
    "prompt": "recolor the car matte black, keep the same person",
    "images": ["data:image/png;base64,..."],
    "loras": [{"name": "krea2_identity_edit_v1_2", "strength": 1.0}],
    "ref_boost": 4.0,
    "grounding_px": 768,
    "fit_mode": "fit",
    "num_inference_steps": 8,
    "guidance_scale": 0.0,
    "mu": 1.15,
    "seed": 42
  }
}
```

Edit fields:

| Field | Default | Notes |
|-------|---------|--------|
| `images` | `[]` | **Required** length 1 for edit (raw base64 or `data:image/...;base64,...`) |
| `width` / `height` | schema 1024 | If **omitted** by the client, size is derived from the source (MP-capped, snap to 16). If present, used as explicit target. |
| `grounding_px` | `768` | Long-side cap for vision TE; `0` = no resize |
| `ref_boost` | `1.0` | Attention boost target→source. **Recommend `4.0` for likeness**; `1.0` is the API default (no dense float bias). |
| `fit_mode` | `"fit"` | `"fit"` (AR-preserving, may be smaller) or `"crop"` (exact target) |
| `num_images` | `1` | Must be **1** for edit |
| `mu` | `1.15` | Always pinned for edit (auto-mu disabled) |

**GQA / `ref_boost`:** when `ref_boost != 1`, the DiT uses a dense float attention bias and expands K/V heads (GQA → full heads) so SDPA has a valid kernel. CUDA backends are FLASH + EFFICIENT only (no MATH — would OOM on full L×L).

**VRAM:** edit sequences are ~2× image tokens (source + target). Same 24 GB class with TE offload; prefer ≤1.5 MP targets. OOM → lower resolution / fewer steps / no CFG.

What is **not** available yet: multi-image refs, “removals → raw” export, Diffusers pipeline rewrite.

Optional `loras` (default `[]`, both modes):

- At most **4** items; names must be unique catalog IDs (exact filename stem, no path/URL/`.safetensors` suffix).
- `type` is optional (default **`lora`**): `"lora"` (A/B or up/down pairs) or `"weight_diff"` (full Linear weight deltas, e.g. `*.diff` projector bypasses).
- `strength` is optional (default **1.0**), finite; range depends on `type`:
  - `lora` → **0.0..2.0**
  - `weight_diff` → **0.0..5.0** (community filter-bypass files often use **3–5**)
  - `0.0` skips that adapter
- Declared `type` must match file contents (no silent auto-detect). No download-by-URL, no TE/VAE LoRA, no `list_loras` API.
- Identity edit expects the identity LoRA on the volume; running edit without it is allowed but quality will be poor.
- Unknown names, type/content mismatch, or invalid shapes fail the job with a safe error (no `refresh_worker`).

Example with a rank LoRA and a projector weight-diff bypass (file pre-placed on volume as `fedor_bypass.safetensors`):

```json
"loras": [
  {"name": "style_v1", "strength": 0.85},
  {"name": "fedor_bypass", "type": "weight_diff", "strength": 4.0}
]
```

What is **not** per-request: which DiT/TE/VAE files (fixed by volume/env), sampler algorithm (official krea-2 Euler flow-matching).

Response:

```json
{
  "output": {
    "images": ["data:image/png;base64,..."],
    "image_url": "data:image/png;base64,...",
    "seed": 42,
    "width": 1024,
    "height": 1024,
    "type": "image_edit",
    "grounding_px": 768,
    "ref_boost": 4.0,
    "fit_mode": "fit",
    "loras": [
      {"name": "krea2_identity_edit_v1_2", "strength": 1.0, "type": "lora"}
    ]
  }
}
```

### curl

```bash
export RUNPOD_API_KEY=rpa_...
export ENDPOINT_ID=...

curl -sS -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d @workers/krea2/test_input.json
```

## Local image build

From monorepo root (same as RunPod: context `.`, Dockerfile under `workers/krea2/`):

```bash
docker build -f workers/krea2/Dockerfile -t runpod-krea2:latest .
```

You still need GPU + `MODEL_DIR` mount to actually generate.

## Design notes

- **VRAM (24 GB):** TE + DiT + VAE load on GPU; after prompt encode the text encoder is offloaded to **CPU** and stays there until the next encode so DiT sampling and VAE decode fit (all-resident OOMs at decode with ~2–3 GiB free).
- **FP8 DiT (default):** weights stay `float8_e4m3fn` where quantized; Linear layers cast to bf16 on the fly so VRAM stays closer to ~12 GB for the transformer.
- **INT8 ConvRot DiT (`DIT_QUANT=int8_convrot`):** stock Comfy `int8_tensorwise` layout (`weight` int8 + `weight_scale` + `comfy_quant`); online Hadamard on activations when marker has `convrot`; CUDA uses `torch._int_mm` for larger token batches.
- **VAE:** prefer clean HF Diffusers weights; do not overlay incompatible Comfy key names with `strict=False`.
- **Runtime LoRA:** up to four pre-placed adapters applied only during DiT denoise (base FP8/INT8 weights unchanged; LoRA delta in compute dtype); released before VAE decode.
- **Identity edit:** grounded multimodal TE (`mm_processor`, separate from text tokenizer) + source VAE tokens with RoPE frame split; optional `ref_boost` dense bias with GQA-safe expand.
- **Not in MVP:** Comfy workflows, multi-ref edit, removals→raw, baking 18 GB into the image layer.

## License

- Third-party licenses and attribution are documented in [`NOTICE`](NOTICE).
- Vendored Krea 2 inference code is licensed under
  [Apache-2.0](LICENSES/KREA-2-APACHE-2.0.txt).
- Portions adapted from RunPod `worker-sdxl` are licensed under
  [MIT](LICENSES/RUNPOD-WORKER-SDXL-MIT.txt).
- These third-party license files do not grant a license to the original code
  in this repository.
- Model weights are not included. Krea 2 weights and derivatives are governed
  by the [Krea 2 Community License](https://www.krea.ai/krea-2-licensing).
