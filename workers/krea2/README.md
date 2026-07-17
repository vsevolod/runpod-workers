# RunPod worker — Krea 2 Turbo FP8

Thin serverless worker: `prompt` → PNG (base64). No ComfyUI.

| | |
|--|--|
| **Model** | [AlperKTS/Krea2_FP8](https://huggingface.co/AlperKTS/Krea2_FP8) DiT + official [krea-ai/krea-2](https://github.com/krea-ai/krea-2) sampler |
| **Text encoder** | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (HF, bf16) |
| **VAE** | Clean Diffusers VAE from [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) (`vae/`); local Comfy single-file is tried first and **ignored** if keys do not match |
| **GPU** | **24 GB** class (TE offload after encode; all-resident OOMs at VAE decode) |
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
└── krea2_infer/            # vendored krea-2 + FP8 loader
```

## Network volume

Mount the same datacenter volume on the endpoint.

### Disk budget (this worker, not Comfy)

| Component | Source | ~size |
|-----------|--------|------:|
| DiT Turbo FP8 | `krea2_turbo_fp8.safetensors` (AlperKTS) | ~12 GB |
| Text encoder | HF `Qwen/Qwen3-VL-4B-Instruct` (bf16 snapshot) | ~8–10 GB |
| VAE | HF `Qwen/Qwen-Image` `vae/` (preferred); optional local single-file | ~0.3 GB |
| **Total** | | **~20–23 GB** |
| Recommended volume | models + HF cache slack (+ future LoRA) | **≥25–30 GB** |

`≥40 GB` is only “plenty of headroom”, not a hard requirement.

```text
/runpod-volume/krea2/
  krea2_turbo_fp8.safetensors     # required (~12 GB)
  qwen_image_vae.safetensors      # optional; ignored if not Diffusers-compatible
/runpod-volume/hf/                # HF_HOME: TE + clean VAE cache
```

Bootstrap on a Pod with the volume attached:

```bash
pip install huggingface-hub hf_transfer
export HF_TOKEN=hf_...   # if needed
export HF_HOME=/runpod-volume/hf
python download_weights.py --output /runpod-volume/krea2 --hf-home /runpod-volume/hf
```

`download_weights.py` also snapshots `Qwen/Qwen-Image` `vae/*` into `HF_HOME` (clean Diffusers VAE). Comfy-format `qwen_image_vae.safetensors` from AlperKTS is optional and often **not** loadable via Diffusers `from_single_file`.

### Comfy FP8 TE is not used

| File | Used here? |
|------|------------|
| `krea2_turbo_fp8.safetensors` | **Yes** (DiT) |
| `qwen_image_vae.safetensors` | Optional try; else clean HF VAE |
| `qwen3vl_4b_fp8_scaled.safetensors` (~5.2 GB) | **No** — ComfyUI `text_encoders/` format |

Text encoding follows **official krea-2**: `Qwen3VLForConditionalGeneration` from Hugging Face. That is larger on disk/VRAM than Comfy’s FP8 TE, but needs no Comfy graph or scaled-FP8 TE loader. Cache the HF model on the volume so cold starts do not re-download it.

## Endpoint env

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_DIR` | `/runpod-volume/krea2` | DiT / optional VAE files |
| `DIT_PATH` | auto under `MODEL_DIR` | Override DiT safetensors path |
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

Request:

```json
{
  "input": {
    "prompt": "a fox walking in the snow",
    "width": 1024,
    "height": 1024,
    "seed": 42,
    "num_inference_steps": 8,
    "guidance_scale": 0.0,
    "mu": 1.15
  }
}
```

Turbo defaults: **8 steps**, **CFG 0**, **mu 1.15**. Width/height multiples of **16** (1024–2048).

These map to Comfy “node knobs” (sampler steps/cfg, latent size, seed). What is **not** per-request today: which DiT/TE/VAE files (fixed by volume/env), sampler algorithm (official krea-2 Euler flow-matching), LoRA.

Response:

```json
{
  "output": {
    "images": ["data:image/png;base64,..."],
    "seed": 42,
    "width": 1024,
    "height": 1024
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
- **FP8 DiT:** weights stay `float8_e4m3fn` where quantized; Linear layers cast to bf16 on the fly so VRAM stays closer to ~12 GB for the transformer.
- **VAE:** prefer clean HF Diffusers weights; do not overlay incompatible Comfy key names with `strict=False`.
- **Not in MVP:** LoRA, Comfy workflows, baking 18 GB into the image layer.

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
