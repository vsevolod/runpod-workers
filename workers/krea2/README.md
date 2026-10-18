# RunPod worker — Krea 2 Turbo (FP8 / INT8 ConvRot)

Thin serverless worker: text-to-image **and** Identity/General Edit V2 via dual
conditioning. No ComfyUI.

| | |
|--|--|
| **Model (default)** | [AlperKTS/Krea2_FP8](https://huggingface.co/AlperKTS/Krea2_FP8) DiT + official [krea-ai/krea-2](https://github.com/krea-ai/krea-2) sampler |
| **Model (optional)** | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) `krea2_turbo_int8_convrot.safetensors` via `DIT_QUANT=int8_convrot` |
| **Text encoder** | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (HF, bf16) |
| **VAE** | Clean Diffusers VAE from [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) (`vae/`); local Comfy single-file is tried first and **ignored** if keys do not match |
| **GPU** | **24 GB** class (DiT/VAE offloaded during text encoding; TE offloaded during denoise/decode). INT8 path is preferred on **RTX 30xx (Ampere)** |
| **Template patterns** | [worker-sdxl](https://github.com/runpod-workers/worker-sdxl) |

## Layout

```
workers/krea2/
├── Dockerfile              # Dockerfile path for RunPod; see build context below
├── handler.py              # RunPod entry + ModelHandler load-once
├── schemas.py
├── download_weights.py     # startup download (not baked into image)
├── start.sh                # shared bootstrap with overlapping downloads
├── test_input.json
├── requirements.txt
└── krea2_infer/            # vendored krea-2 + FP8 / INT8 loaders
```

## Model storage and startup

Hub transfers, cache/token setup and timing use the shared
[`model_download` module](../shared/model_download.py), copied into the image.
`download_weights.py` keeps Krea's FP8/INT8 selection and TE/VAE preparation.

Use an **80 GB container disk**. On each fresh worker, `start.sh` starts the
shared bootstrap with a separate CPU-only downloader. ML imports run immediately
with `LOCAL_FILES_ONLY=1`. The download order is DiT → clean Diffusers VAE → text
encoder → Identity Edit V2 + General Edit V2. DiT files go to `/models/krea2`;
HF snapshots go to `/models/hf`. Both edit adapters (~1.83 GB each) are downloaded
to container disk and exposed under `$MODEL_DIR/edit_loras`.

The loader waits for each component immediately before loading it, so DiT can
load into VRAM while VAE/encoder downloads continue. Each repo is pinned to one
commit for the session, and the VAE/encoder loaders receive the exact local
snapshot paths. Loading into VRAM remains sequential. No requests are accepted
until all required components, including both V2 adapters, are ready and the downloader exits successfully.
A failed transfer or model load aborts startup and reaps the downloader.
New containers download again because their storage is temporary.

The startup download skips the optional Comfy VAE single-file and uses the HF
VAE cache. Additional LoRAs remain on the attached volume at
`/runpod-volume/krea2/loras`. Logs distinguish `model_download`, `component_ready`,
`wait_component` and `component_load`. Include the entire startup and first
generation when comparing speed; overlapping stages must not be added together.

Existing endpoint environment values override Dockerfile defaults. After
publishing the new image, update the endpoint explicitly:

```text
MODEL_DIR=/models/krea2
HF_HOME=/models/hf
LORA_DIR=/runpod-volume/krea2/loras
DIT_QUANT=int8_convrot
```

Keep your current `DIT_QUANT` selection (`int8_convrot` is used by the existing
`krea2-workers` endpoint; new images still default to `fp8`). Remove or update
any explicit `DIT_PATH` / `VAE_PATH` pointing at old remote weights. The entrypoint
sets `LOCAL_FILES_ONLY=1` and waits for each required component; no pre-populated model volume
is required.

The endpoint's Docker start command must use the image's default `CMD` or
`sh /app/start.sh`. Remove old commands that launch `handler.py` directly after
the new image is active; see [startup command migration](../../README.md#existing-endpoint-startup-commands).

### Optional persistent snapshot

The downloader can also bootstrap a volume manually. Regional network volumes
must be in the same datacenter as the endpoint; Global Volumes are not tied to
a datacenter. The default startup downloads directly to container disk.

### Disk budget (this worker, not Comfy)

| Component | Source | ~size |
|-----------|--------|------:|
| DiT Turbo FP8 | `krea2_turbo_fp8.safetensors` (AlperKTS) | ~12 GB |
| DiT Turbo INT8 ConvRot (optional) | `krea2_turbo_int8_convrot.safetensors` (Comfy-Org) | ~13.5 GB |
| Text encoder | HF `Qwen/Qwen3-VL-4B-Instruct` (bf16 snapshot) | ~8–10 GB |
| VAE | HF `Qwen/Qwen-Image` `vae/` (preferred); optional local single-file | ~0.3 GB |
| Edit V2 LoRAs | Identity + General, downloaded automatically | ~3.66 GB |
| **Total (FP8 path)** | | **~24–27 GB** |
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

Identity Edit V2 and General Edit V2 are built in: no volume is required for them.
Downloads are pinned to `conradlocke/krea2-identity-edit` revision
`f33b26291c640e8ec50b0da420bfee3c9a660daa`. They take priority over same-named
files on the volume. There is no v1.2 fallback.

Place additional Krea 2 adapter `.safetensors` files in `loras/` (or `$LORA_DIR`): standard
rank LoRA and/or full `weight_diff` files (e.g. Fedor-style
`diffusion_model.txtfusion.projector.diff`). The worker scans **only top-level**
`*.safetensors` stems into an allowlist at startup and refreshes it on requests
with adapters; files are copied locally and loaded only when selected for a job.
Request `type` must match each file’s tensor layout. Warm workers discover added
or removed adapters without a restart.

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

Text encoding follows **official krea-2**: `Qwen3VLForConditionalGeneration` from Hugging Face. That is larger on disk/VRAM than Comfy’s FP8 TE, but needs no Comfy graph or scaled-FP8 TE loader. The startup downloader warms its local HF cache before inference, honoring `TEXT_ENCODER_ID` when overridden.

### DiT quant modes (`DIT_QUANT`)

| `LORA_CACHE_DIR` | `$MODEL_DIR/lora_cache` | Local container-disk cache of requested adapters |
| `DIT_QUANT` | DiT file (auto under `MODEL_DIR`) | Compute path |
|-------------|-----------------------------------|--------------|
| `fp8` (default / unset) | `krea2_turbo_fp8.safetensors` | FP8 storage, cast to bf16 on Linear forward (current production) |
| `int8_convrot` (aliases: `int8`, `int8_tensorwise`) | `krea2_turbo_int8_convrot.safetensors` | int8_tensorwise + online ConvRot; CUDA `torch._int_mm` W8A8 when batch is large enough, else dequant fallback |

Mode is fixed at worker start (FlashBoot-friendly). Mismatched file vs mode fails fast. TE / VAE / API are shared.

**Why INT8 on 3090:** Ampere has INT8 Tensor Cores but not FP8. Native INT8 GEMM avoids the FP8→BF16 cast tax. Speedup only applies when the INT8 path is active — plain dequant of an int8 file under `DIT_QUANT=fp8` is rejected.

## Endpoint env

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_DIR` | `/models/krea2` (Dockerfile) | DiT files downloaded before DiT loading |
| `LORA_DIR` | `/runpod-volume/krea2/loras` | Source directory of LoRA `.safetensors`; refreshed on requests with adapters |
| `DIT_QUANT` | `fp8` | `fp8` or `int8_convrot` — selects DiT load + Linear runtime |
| `DIT_PATH` | auto under `MODEL_DIR` | Override DiT safetensors path (must match `DIT_QUANT`) |
| `VAE_PATH` | auto under `MODEL_DIR` | Override VAE safetensors path |
| `TEXT_ENCODER_ID` | `Qwen/Qwen3-VL-4B-Instruct` | HF id or local snapshot path |
| `HF_HOME` | `/models/hf` (Dockerfile) | Local cache for the text encoder and clean VAE |
| `LOCAL_FILES_ONLY` | `1` (set by start.sh) | Inference uses only completed local files/caches |
| `BUCKET_ENDPOINT_URL` | unset | If set, upload via RunPod S3 helpers instead of base64 |
| `TORCH_COMPILE_DISABLE` | `1` (Dockerfile) | Skip torch.compile / Inductor (no gcc in image) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` (Dockerfile) | Reduce CUDA allocator fragmentation |

## Deploy (RunPod)

Repo may be **public or private** (connect GitHub in RunPod for private). Do not commit tokens or weights.

1. Build through the **Build Runpod workers** GitHub Actions workflow, then use
   **Serverless → New Endpoint → Import from Docker Registry**, or update the
   existing endpoint's template while retaining its ID. The image is
   `ghcr.io/vsevolod/runpod-workers-krea2:<commit-sha>`.
2. Set **Container disk = 80 GB** and the local model/cache paths above. The worker
   downloads its own base weights at startup.
3. Attach a volume only if using additional LoRAs, preserving their `LORA_DIR`.

### Build settings (monorepo)

| Setting | Value | Notes |
|---------|-------|--------|
| **Branch** | `master` (this repo) | Not `main` unless you rename |
| **Dockerfile path** | `workers/krea2/Dockerfile` | Path to the file only |
| **Build context** | `.` | **Repo root** — required because Dockerfile does `COPY workers/krea2/...` |

Wrong build context (e.g. `workers/krea2`) makes `COPY workers/krea2/...` fail during image build. Context does **not** control *when* a build starts; it only controls how Docker resolves paths.

Other endpoint knobs:

- GPU: **24 GB** class (A5000 / 3090 / 4090 / L4, etc.)
- Attach a volume if needed for additional LoRAs
- Container disk: **80 GB**
- FlashBoot: on
- Active workers: `0` (or `1` for warm)
- Env: `MODEL_DIR=/models/krea2`, `HF_HOME=/models/hf`; preserve `DIT_QUANT` and the existing `LORA_DIR`

### Updating the endpoint after code changes

Worker changes pushed to `master` trigger the GitHub Actions build and GHCR
publication. For a manual build:

```bash
gh workflow run build-workers.yml --ref master -f publish=true
gh run list --workflow build-workers.yml
```

After the workflow succeeds, update the Runpod endpoint to the exact published
image tag/digest and verify a GPU request. GitHub Actions does not automatically
change the endpoint template. See the shared [deployment instructions](../../README.md#qwen-image-and-krea2-build-on-github-run-on-runpod).

## API

The worker uses [`shared/api`](../shared/api/README.md) together with Qwen.
See that document for the complete contract, defaults, responses and errors.
New clients put `mu`, `grounding_px`, reference controls and `fit_mode` inside
`model_options`; the flat examples below remain supported as compatibility aliases.
`reference_megapixels` is a common **top-level** field, accepting a number or
`"max"` (default). Max selects the active FP8/INT8 profile and total CUDA VRAM
in the shared [table](../shared/megapixels.py). The 48 GB tier selects 4 MP,
measured on L40S with Identity Edit V2, two square references and CFG off;
other tiers remain 1 MP placeholders. See the
[benchmark procedure and results](../../docs/benchmarks/image-megapixels.md).
Explicit numbers bypass the table. Both the selected number and detected VRAM
are reported in the common response; see the [shared API](../shared/api/README.md#shared-megapixel-budget).
The common `sampler` field accepts only `"euler"` on Krea. Both `"lora"` and
`"weight_diff"` entries remain supported in the top-level `loras` list.

### `type` field

| `type` | Meaning | `images` |
|--------|---------|----------|
| `image_generate` (default) | Text-to-image | must be empty / omitted |
| `image_edit` | Identity edit (1–2 refs + instruction) | **1 or 2** base64 / data-URL entries |

`images` is always an **array**. Edit requires length **1 or 2** (hard max 2).

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

### Identity / General Edit V2 (`image_edit`)

Dual conditioning (not a plain LoRA attach):

1. **Grounded Qwen3-VL encode** — instruction + **1 or 2** source images through the vision TE (`mm_processor`).
2. **Source VAE tokens** — each ref independently fit/crop → encode → RoPE frames `1..N` concatenated with target noise (frame=`0`).

Reference implementation: [ComfyUI-Krea2Edit v2.0.0](https://github.com/lbouaraba/comfyui-krea2edit/releases/tag/v2.0.0),
commit `13ef60f20828949becada815d3cbc371eb5ccaab`. The thin worker implements
crop-to-grid fitting, fractional RoPE centering and per-reference placement.
Select one built-in adapter through `loras`:

```text
krea2_identity_edit_v2   # restage a person while preserving likeness
krea2_general_edit_v2    # remove, recolor, relight, backgrounds, objects, text, outfits
```

Both are downloaded automatically at startup. Use one per task. Loading both
at strength 1.0 produces noise; the author's optional blend is about 0.6 each.
The API does not insert an adapter when `loras` is empty.

Recommended Turbo presets (8 steps, worker guidance 0, grounding 768):

| Task | Adapter | Last/only ref boost | Placement |
|---|---|---:|---|
| Restage a person | Identity V2 | 2 | `beside` |
| Keep portrait framing | Identity V2 | 1–2 | `aligned` |
| Put a person into a scene | Identity V2 | person 2–3, scene 1 | person `beside`, scene `aligned` |
| Instruction edit | General V2 | 4 | `aligned` |

The CLI selects these defaults with `--edit-model identity` (default) or
`--edit-model general`; `--lora` overrides the preset's adapter selection.

```bash
python scripts/krea2_image.py -i person.jpg --prompt "Full body portrait in a blue suit" --seed 42
python scripts/krea2_image.py -i scene.jpg --edit-model general --prompt "Remove the car" --seed 42
```

**Single ref:**

```json
{
  "input": {
    "type": "image_edit",
    "prompt": "recolor the car matte black, keep everything else unchanged",
    "images": ["data:image/png;base64,..."],
    "loras": [{"name": "krea2_general_edit_v2", "strength": 1.0}],
    "ref_boost": 4.0,
    "ref_position": "aligned",
    "grounding_px": 768,
    "fit_mode": "fit",
    "num_inference_steps": 8,
    "guidance_scale": 0.0,
    "mu": 1.15,
    "seed": 42
  }
}
```

**Two refs** (same pipeline — multi-identity, compose, or scene+subject). Order matters for scene+subject: **scene first, subject second**.

```json
{
  "input": {
    "type": "image_edit",
    "prompt": "create a photo of this person next to the tractor",
    "images": [
      "data:image/png;base64,...",
      "data:image/jpeg;base64,..."
    ],
    "loras": [{"name": "krea2_identity_edit_v2", "strength": 1.0}],
    "ref_boost": 2.0,
    "ref_boost_a": 1.0,
    "ref_position": "beside",
    "ref_position_a": "aligned",
    "fit_mode": "fit",
    "seed": 42
  }
}
```

Edit fields:

| Field | Default | Notes |
|-------|---------|--------|
| `images` | `[]` | **Required** length **1 or 2** (raw base64 or `data:image/...;base64,...`) |
| `width` / `height` | schema 1024 | If **both omitted**, canvas uses the original **`images[0]`** aspect at the full shared MP budget, rounded down to 16px, including upscaling small sources. Explicit dimensions take priority. `images[1]` never sets canvas. |
| `grounding_px` | `768` | Long-side cap for vision TE **per** image; `0` = no resize |
| `ref_boost` | `1.0` | Attention boost toward the **last/only** ref: person in scene+person workflows. Recommend 2 for Identity, 4 for General. |
| `ref_boost_a` | `1.0` | Independent boost toward the **first** ref in two-image workflows (scene); ignored for one ref. |
| `ref_position` | `"aligned"` | Last/only ref placement: `aligned`, `beside`, `bag_of_features` (experimental). |
| `ref_position_a` | `"aligned"` | First ref placement in two-image workflows; ignored for one ref. |
| `fit_mode` | `"fit"` | `"fit"` (AR-preserving, may be smaller) or `"crop"` (exact target); applied to **all** refs |
| `num_images` | `1` | Must be **1** for edit (output count, not ref count) |
| `mu` | `1.15` | Always pinned for edit (auto-mu disabled) |

**GQA / reference boosts:** when either active reference boost differs from 1, the DiT uses a dense float attention bias and expands K/V heads (GQA → full heads) so SDPA has a valid kernel. CUDA backends are FLASH + EFFICIENT only (no MATH — would OOM on full L×L).

**VRAM:** 1-ref edit ≈ ~2× image tokens (source + target); **2 refs ≈ ~3×**. Same 24 GB class with TE offload; prefer ≤1.5 MP targets. OOM → lower resolution / fewer steps / no CFG / single ref.

Before conditioning, input references share the resolved `reference_megapixels`
budget equally and are downscaled on CPU, preserving aspect and never enlarged.
Their unused shares are not redistributed. The source-pixel fit step still
resamples each reference to the training-matched target grid, aligned to 16px;
the input budget therefore does not cap the combined VAE token count.
The automatic output uses the full budget regardless
of reference count. The CLI accepts `--reference-megapixels max` or a number.

What is **not** available yet: N>2 refs, Raw-model editing, Diffusers pipeline rewrite.
The author's Raw/20-step/CFG-3 preset for difficult removals requires a different
base checkpoint; this worker remains Turbo. Style-only dual-ref is not a quality guarantee.

Optional `loras` (default `[]`, both modes):

- At most **4** items; names must be unique catalog IDs (exact filename stem, no path/URL/`.safetensors` suffix).
- `type` is optional (default **`lora`**): `"lora"` (A/B or up/down pairs) or `"weight_diff"` (full Linear weight deltas, e.g. `*.diff` projector bypasses).
- `strength` is optional (default **1.0**), finite; range depends on `type`:
  - `lora` → **0.0..2.0**
  - `weight_diff` → **0.0..5.0** (community filter-bypass files often use **3–5**)
  - `0.0` skips that adapter
- Declared `type` must match file contents (no silent auto-detect). No download-by-URL, no TE/VAE LoRA, no `list_loras` API.
- Edit expects one selected V2 adapter; both are available locally after startup.
- Unknown names, type/content mismatch, or invalid shapes fail the job with a safe error (no `refresh_worker`).

Each request copies only its nonzero-strength adapters from `LORA_DIR` to
`LORA_CACHE_DIR` on container disk. Copying overlaps text/reference conditioning;
loading and GPU activation wait for complete local files. A warm worker checks the
cache again on every request, restores missing/incomplete files, and refreshes
changed source files (size/mtime/ctime). Newly uploaded adapters are discovered
without restarting. Copies publish atomically; a failed transfer is retryable and
never falls back to network-backed mmap. No full-catalog download occurs.
`stage=lora_download` logs show copies, local paths, and cache hits. Keep enough
container disk for the base models plus the adapters used by this worker; the cache
persists until the container is destroyed and has no automatic eviction.

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
    "seed": 42,
    "width": 1024,
    "height": 1024,
    "type": "image_edit",
    "grounding_px": 768,
    "ref_boost": 4.0,
    "fit_mode": "fit",
    "num_refs": 1,
    "loras": [
      {"name": "krea2_identity_edit_v2", "strength": 1.0, "type": "lora"}
    ]
  }
}
```

Images are returned only in the `images` array; there is no `image_url` alias.
With `BUCKET_ENDPOINT_URL` set, its entries are URLs instead of base64 data URLs.
Edit responses keep top-level `grounding_px` / `ref_boost` / `fit_mode` and add top-level **`num_refs`** (`1` or `2`).

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

Generation needs a GPU and enough container disk for the startup downloads.

## Design notes

- **Profiling:** stage logs cover adapter preparation/copy, encoder/model transfers, each denoise step and VAE work. Request logs include numeric settings and counts, without prompt/image contents. Set `KREA_PROFILE_CUDA=1` temporarily for synchronized GPU timings; the default `0` avoids synchronization overhead, so GPU stage timings can include queued work in later phases. Profiling does not change the endpoint execution timeout.
- **VRAM (24 GB):** the text encoder starts on **CPU**. During text/grounded encoding, DiT and VAE move to CPU before the encoder moves to GPU. After positive/negative conditioning, the encoder returns to CPU, then DiT/VAE return to GPU for denoise/decode. Device transfers preserve quantized weight dtypes. This reduces peak VRAM at the cost of RAM use and CPU/GPU transfer time. If encoding fails, the encoder is offloaded and DiT/VAE stay on CPU while the handler reports the error.
- **FP8 DiT (default):** weights stay `float8_e4m3fn` where quantized; Linear layers cast to bf16 on the fly so VRAM stays closer to ~12 GB for the transformer.
- **INT8 ConvRot DiT (`DIT_QUANT=int8_convrot`):** stock Comfy `int8_tensorwise` layout (`weight` int8 + `weight_scale` + `comfy_quant`); online Hadamard on activations when marker has `convrot`; CUDA uses `torch._int_mm` for larger token batches.
- **VAE:** prefer clean HF Diffusers weights; do not overlay incompatible Comfy key names with `strict=False`.
- **Runtime LoRA:** up to four selected built-in or volume adapters applied only during DiT denoise (base FP8/INT8 weights unchanged; LoRA delta in compute dtype); released before VAE decode.
- **Edit V2:** grounded multimodal TE (`mm_processor`, separate from text tokenizer) + 1–2 source VAE streams; per-reference RoPE placement and boost with GQA-safe attention.
- **Not in MVP:** Comfy workflows, N>2 refs, removals→raw, baking 18 GB into the image layer.

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
- Identity Edit V2 and General Edit V2 additionally use the author's
  [V2 model license](LICENSES/KREA-2-EDIT-V2.md).
