# Runpod worker — Qwen-Image-2.1 Base / Turbo

Thin serverless worker: text-to-image and multi-reference edit through
Diffusers `QwenImage21Pipeline`. No ComfyUI. Same handler shape as
[`workers/krea2`](../krea2/) (base64 or S3, FlashBoot). One endpoint selects
Base or Turbo per request and keeps one pipeline resident at a time.

| | |
|--|--|
| **Models** | [Base](https://huggingface.co/Qwen/Qwen-Image-2.1): default 40 steps, Euler/simple; [Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo): default 8 checkpoint-configured steps |
| **Pipeline** | `QwenImage21Pipeline` from Diffusers, pinned to commit `da1d382` (PR [#14950](https://github.com/huggingface/diffusers/pull/14950)), with the native-reference-size patch |
| **Text encoder** | Qwen3-VL 8B, loaded with the snapshot |
| **VAE** | [Texture-Fix VAE](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1), a compatible 64-channel RGBA `AutoencoderKLQwenImage21`. Set `QWEN_VAE=original` to use the checkpoint's VAE. The Qwen-Image / Krea 2 VAE is not interchangeable |
| **GPU** | **24 GB** with `OFFLOAD=cpu` at 1024. Native 2048 wants a larger GPU or `OFFLOAD=sequential` |
| **License** | [Qwen Research License](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/LICENSE). Research and evaluation only, unless you have a commercial agreement with Qwen |

## Layout

```
workers/qwen_image_21/
├── Dockerfile
├── handler.py
├── schemas.py
├── download_weights.py
├── start.sh
├── test_input.json
├── requirements.txt
└── qwen_image_21_infer/
```

## Model storage and startup

Hub transfers, cache/token setup and timing use the shared
[`model_download` module](../shared/model_download.py), copied into the image.
`download_weights.py` keeps Qwen's model selection and variant-specific validation.

Use a **120 GB container disk** to keep both snapshots, shared VAE, image/runtime
and requested LoRAs. The model weights use about 65 GB with separate text-encoder
copies and one shared Texture-Fix VAE. Each variant alone uses about 33 GB.

Startup imports the handler without preselecting a model. The first request's
`model_variant` chooses the checkpoint before a separate CPU-only downloader
starts. Imports/probing overlap its transfers. Only that variant is downloaded;
a first Base request never preloads Turbo. Files go to
`/models/qwen_image_21_turbo` or `/models/qwen_image_21_base`.

On a warm request, complete local manifests are checked without Hub access.
Missing/truncated files trigger repair before inference, including on the already
selected model. The manifest records relative file names and byte sizes; it is
published after a successful download. Each selected snapshot is validated before
loading. Existing mmap and CPU-offload behavior is preserved.

Generation and model changes share one lock. Repeated requests for the selected
variant reuse its pipeline. Switching variants removes offload hooks, releases
the old transformer/text encoder and adapter manager, collects Python references,
and empties unused CUDA allocations before loading the replacement from disk.
The shared Texture-Fix VAE is reused in memory while its local files are complete.
Text encoders remain variant-specific; identical file sizes do not establish
identical weights. A failed switch does not fall back to the previous variant.

Downloads use a fixed repo commit for the session and the HF cache stays local
at `/models/hf`. `LOCAL_FILES_ONLY=1` prevents implicit runtime downloads.
A failed download or model load fails the request and reaps the downloader. An
interrupted transfer can resume in the same container.

By default, the downloader skips the unused stock `vae/` and downloads only
the Diffusers config and weights of `madebyollin/texture-fix-vae-for-qwen-image-2.1`
into `QWEN_VAE_DIR` (default `/models/qwen_image_21_shared/texture_fix_vae`).
It publishes the selected model as ready after
both downloads complete. The loader supplies this VAE to `from_pretrained`
before CPU/GPU placement, preserving bf16, mmap and CPU offload. No runtime
Hub access is needed with `LOCAL_FILES_ONLY=1`.

Texture-Fix is an unofficial VAE finetune intended to improve fine textures
and checkerboard artifacts. Its author reports improved perceptual quality
with slightly lower reconstruction accuracy; improvement on our edits still
needs a GPU comparison. It does not change the Turbo transformer, text encoder,
LoRAs or guidance. At eight steps, Turbo keeps its checkpoint's `sample_sigmas`.

For an A/B comparison, set `QWEN_VAE=original` and replace the worker. Its first
request will download the selected checkpoint's original VAE into `vae/`.
Original VAEs stay variant-specific. Restore `QWEN_VAE=texture-fix` and
replace the worker to switch back. Use the same prompt, references, seed,
resolution and LoRAs in both runs. The choice is fixed for the worker lifetime.

The handler memory-maps the **local** weights (`QWEN_DISABLE_MMAP=0`). Model
download time and the final wait are logged as `model_download` and
`wait_component`; include the entire interval through the first completed image
when comparing cold starts. A new worker downloads
again because container storage is temporary.

Existing endpoint env overrides must be updated explicitly:

```text
MODEL_DIR=/models/qwen_image_21_turbo
BASE_MODEL_DIR=/models/qwen_image_21_base
QWEN_VAE_DIR=/models/qwen_image_21_shared/texture_fix_vae
HF_HOME=/models/hf
QWEN_DISABLE_MMAP=0
LORA_DIR=/runpod-volume/qwen_image_21_turbo/loras
```

Keep the attached volume for additional LoRAs. If your adapters are in a
different directory, preserve that path in `LORA_DIR`.

### Manual prefetch on container disk

The same downloader can prepare either variant without running inference.
Pass the shared VAE path for both commands so it is downloaded once:

```bash
pip install huggingface-hub hf_transfer
export HF_TOKEN=hf_...    # accept the model license on the Hub first
python workers/qwen_image_21/download_weights.py \
  --model-variant turbo --output /models/qwen_image_21_turbo \
  --vae-dir /models/qwen_image_21_shared/texture_fix_vae
python workers/qwen_image_21/download_weights.py \
  --model-variant base --output /models/qwen_image_21_base \
  --vae-dir /models/qwen_image_21_shared/texture_fix_vae
```

Snapshots use separate directories. Turbo requires its valid saved eight-step
grid; Base rejects a saved Turbo grid. The downloader refuses to overwrite a
directory containing the other variant. Complete snapshots are reused, and
missing or truncated manifest-listed files are downloaded again.

Container disk is temporary: manual prefetch survives only while that container
exists. The attached Global Volume remains the source for LoRAs. The default
model downloader uses Hugging Face; it does not implicitly load an old snapshot
from the attached volume.

## Migrating from the Viggle worker

1. Rebuild the image with the pinned Diffusers dependency. Keep `MODEL_DIR`,
   `BASE_MODEL_DIR`, `QWEN_VAE_DIR` and `HF_HOME` on local container disk, and
   allow 120 GB for both variants. Existing environment overrides take precedence
   over Dockerfile defaults.
2. If reusing your existing style/character adapters, set `LORA_DIR` to
   `/runpod-volume/qwen_image_21/loras`. Obsolete Viggle accelerators are ignored.
   Layer compatibility is validated; visual quality with Turbo needs a GPU check.
3. Default steps are **8 for Turbo / 40 for Base**; overrides accept 1..200.
   Add `model_variant` to
   select Base. Existing clients without that field keep Turbo behavior.
   Check generation, multi-reference edits, RGBA output and a request with then
   without an additional LoRA on the deployed GPU worker.

## Endpoint env

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_DIR` | `/models/qwen_image_21_turbo` (Dockerfile) | Snapshot validated before model loading |
| `MODEL_ID` | `Qwen/Qwen-Image-2.1-Turbo` | Turbo Hub repo, fetched only when requested |
| `BASE_MODEL_DIR` | sibling `qwen_image_21_base` beside `MODEL_DIR` | Separate local Base snapshot |
| `BASE_MODEL_ID` | `Qwen/Qwen-Image-2.1` | Base Hub repo, fetched only when requested |
| `QWEN_VAE_DIR` | `/models/qwen_image_21_shared/texture_fix_vae` | Shared local Texture-Fix config and weights |
| `QWEN_VAE` | `texture-fix` | Selected VAE: `texture-fix` or `original`; restart the worker to change it |
| `LORA_DIR` | `/runpod-volume/qwen_image_21_turbo/loras` (Dockerfile) | Additional adapters on the attached volume; preserve your existing adapter path |
| `LORA_CACHE_DIR` | `$MODEL_DIR/lora_cache` | Local container-disk cache of requested adapters |
| `OFFLOAD` | `cpu` | `cpu` (model offload, 24 GB), `sequential` (tighter), `none` (resident on CUDA) |
| `LOCAL_FILES_ONLY` | `1` (set by start.sh) | Inference loads only the completed local snapshot |
| `HF_HOME` | `/models/hf` (Dockerfile) | Local HF cache; keep off the remote volume |
| `BUCKET_ENDPOINT_URL` | unset | If set, upload via RunPod S3 helpers instead of base64 |
| `TORCH_COMPILE_DISABLE` | `1` (Dockerfile) | Skip torch.compile |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduce allocator fragmentation |
| `QWEN_STARTUP_TRACE_INTERVAL` | `60` | Dump Python thread stacks every N seconds during model loading; `0` disables dumps |
| `QWEN_DISABLE_MMAP` | `0` | Memory-map base weights; `1` opts into eager loading with higher startup RAM usage |

`OFFLOAD=cpu` keeps one component on GPU at a time
(`text_encoder → transformer → vae`). bf16 weights do not fit together in 24 GB:
the transformer is ~14 GB and Qwen3-VL-8B is ~16 GB. Turbo weights are already
part of the transformer checkpoint; no accelerator LoRA is attached.

### Network-volume loading experiment

`QWEN_DISABLE_MMAP=0` is the default to keep startup RAM usage lower.
Set `QWEN_DISABLE_MMAP=1` only to compare eager loading against mmap on a
worker with sufficient RAM headroom. In this experimental mode the text encoder
is loaded separately with Transformers' `disable_mmap=True`, then passed into the pipeline.
Diffusers loads the transformer and VAE with the same option.
Request-scoped LoRA loading uses its own existing loading path.

Eager loading increases startup RAM usage. A 62 GB container reached about
58 GB before its first request and RunPod reported container OOM during that
request at 1024 with an additional LoRA. CPU offload controls GPU placement;
it does not remove this host-memory loading peak. Compare
the **whole** `stage=startup` duration, not only `stage=placement_none`:
file reading now happens inside `text_encoder_load` and
`pipeline_components_load`, before placement on GPU. Check cgroup memory
metrics as well. The runtime logs `disable_mmap=True` when enabled.

On existing deployments, set `QWEN_DISABLE_MMAP=0` in the endpoint environment
and replace the worker, including any paused FlashBoot worker. Images that
already support this variable do not need a rebuild for the override. Rebuild
the image to pick up the changed default for future deployments.

### Startup diagnostics

At `INFO` level the worker logs `stage=<name> begin/done/failed`, elapsed
seconds, PID, process RSS, and available cgroup memory usage/limit/OOM counters.
CUDA free/total/allocated/reserved memory is included after CUDA initializes.
Runtime logs include library versions, Torch CPU thread count, GPU name,
compute capability and architectures compiled into Torch.

Separate stages cover imports, source resolution, catalog scanning, base model
loading (including `vae_load` for Texture-Fix), sampling-config validation, CUDA probing,
and placement on GPU or CPU offload. Inference and image delivery also have
timings. The last `begin` without a matching `done` identifies the unfinished
stage. No prompts or image contents are logged by these diagnostics.

During startup, thread stacks go to stderr every 60 seconds by default,
including when native loading code holds the Python GIL. Faulthandler labels
these dumps `Timeout`; this is a diagnostic timer and does **not** terminate
the worker. Dumps stop on successful loading or an exception. A process killed
externally (such as SIGKILL/OOM) cannot log its own termination reason; check
RunPod's container exit reason as well.

## Deploy (RunPod)

| Setting | Value |
|---------|--------|
| **Dockerfile path** | `workers/qwen_image_21/Dockerfile` |
| **Build context** | `.` (repo root) |
| **GPU** | 24 GB class with `OFFLOAD=cpu`; 48 GB+ if you set `OFFLOAD=none` and generate at 2048 |
| **Volume** | Optional for additional LoRAs; base weights download to container disk |
| **Container disk** | 120 GB for both variants and LoRAs |

Push worker changes to `master` and wait for the **Build Runpod workers** GitHub
Actions workflow to publish `ghcr.io/vsevolod/runpod-workers-qwen_image_21:<commit-sha>`.
Configure Runpod to use that exact image tag/digest, preserving the endpoint ID
and existing GPU/LoRA settings. See the shared [deployment instructions](../../README.md#qwen-image-and-krea2-build-on-github-run-on-runpod).
Verify that the intended image is active before running a GPU smoke request.
Use the image's default Docker start command or `sh /app/start.sh`; the older
`download_weights.py && handler.py` override bypasses the shared bootstrap.
While the old image is active, use the compatible command documented in
[startup command migration](../../README.md#existing-endpoint-startup-commands).

## API

The worker uses [`shared/api`](../shared/api/README.md) together with Krea.
That document is the complete contract, including common fields, model-specific
options, responses, errors and compatibility aliases. New clients should send
Qwen-specific controls under `"model_options": {"variant": "base"}` and optional
`output_resolution` inside that object. `reference_megapixels` is a common top-level
field accepting a number or `"max"` (default). Existing flat examples below remain
supported as compatibility aliases.

The following fields work for both `image_generate` and `image_edit`, with
either model variant. Effective values are returned in the response and logged.

| Field | Default | Overrides |
|-------|---------|-----------|
| `num_inference_steps` | Turbo: `8`; Base: `40` | Integer in `1..200` |
| `guidance_scale` | `1.0` | Finite number `>= 0`; maps to Diffusers `true_cfg_scale` |
| `sampler` | `"euler"` | `"euler"`, `"heun"`, `"er_sde"` |

`guidance_scale <= 1` disables classifier-free guidance. Values above 1 execute
both conditional and unconditional passes at each evaluation. `negative_prompt`
is optional: if omitted or blank, CFG uses an empty unconditional prompt. Explicit
negative text is preserved. Turbo defaults to CFG off.

`model_variant` accepts `"turbo"` (default) or `"base"` for both generation and edit.
Omitted `num_inference_steps` resolves to **8 for Turbo** or **40 for Base**.
Explicit step counts override these defaults.
Existing clients without `model_variant` retain Turbo behavior.

At **8 steps**, Turbo retains its checkpoint's saved `sample_sigmas` and grid
transforms (`scheduler: "checkpoint"`). For **other Turbo counts**, the request
explicitly supplies a `simple` grid of the requested length. This is an
experimental override of the model's trained eight-step schedule, with no
guarantee of improved quality. Base always uses `scheduler: "simple"`.

The simple grid selects the requested number of sigmas using ComfyUI's
`simple_scheduler` from `ModelSamplingFlux`'s
10000-point table with QwenImage21's default shift **0.69**. Its scheduler is
flow matching with dynamic shifting and terminal stretching disabled because
the supplied grid is already shifted. Diffusers appends terminal zero. These
parameters match the default ComfyUI QwenImage21 Euler/simple combination; custom
Comfy model-sampling patches and bit-identical cross-framework images are outside
this contract. Sampling parity is checked numerically on CPU.

Euler makes one denoiser evaluation per step. Heun makes **2 × steps − 1**
evaluations (predictor/corrector, ending with Euler at zero). ER-SDE uses ComfyUI's
ER-SDE-Solver-3 defaults: `max_stage=3`, `s_noise=1`, and the default noise scaler.
It makes one evaluation per step, with additional stochastic noise drawn from the
request's seeded generator. CFG above 1 doubles these denoiser pass counts.

Schedulers are created per request and discarded afterward, including on errors.
Changing Steps/CFG/Sampler does not reload model weights. ER-SDE is adapted from
[ComfyUI ce1e922](https://github.com/Comfy-Org/ComfyUI/blob/ce1e922d8623bb507180e32a0c01a7d0e29edef7/comfy/k_diffusion/sampling.py)
under [GPL-3.0](LICENSES/COMFYUI-GPL-3.0.txt). The Heun adapter supplies the selected
sigma grid to Diffusers' flow-matching Heun solver; these are not the ordinary
SD/SDXL discrete schedulers.

Width and height must be multiples of **32**, each side in 256..2752, and the
area must fit the largest official preset (`2400×1792`). Recommended sizes:

| Aspect | Size |
|--------|------|
| 1:1 | 2048×2048 |
| 4:3 | 2400×1792 |
| 3:4 | 1792×2400 |
| 3:2 | 2528×1696 |
| 2:3 | 1696×2528 |
| 16:9 | 2752×1536 |
| 9:16 | 1536×2752 |

The schema default canvas is **1024×1024** so a 24 GB worker can take a job
without an immediate OOM. Pass an official size when the GPU has room.

### Text-to-image

```json
{
  "input": {
    "type": "image_generate",
    "prompt": "a fox walking in the snow",
    "width": 1024,
    "height": 1024,
    "seed": 42,
    "num_inference_steps": 8,
    "guidance_scale": 1.0
  }
}
```

`images` must be empty. `num_images` may be 1..4.

### Edit (1–10 references)

```json
{
  "input": {
    "type": "image_edit",
    "prompt": "these three characters sit around a campfire",
    "images": ["data:image/png;base64,..."],
    "num_inference_steps": 8
  }
}
```

Top-level `reference_megapixels` is the **total** reference budget (default
`"max"`). The worker selects a number from the shared model/VRAM
[table](../shared/megapixels.py). The 48 GB tier selects **4 MP** for Base and
Turbo, measured on L40S with three square references, Lenovo LoRA and CFG 1;
other tiers remain **1 MP** placeholders. See the
[benchmark procedure and results](../../docs/benchmarks/image-megapixels.md).
Explicit numbers bypass the table. The old nested input remains
accepted for compatibility. Each reference is downscaled on CPU to at most this budget divided
by the number of references, before both the vision/text encoder and VAE.
One megapixel is 1,000,000 pixels. Aspect ratio is preserved subject to rounding
dimensions down to multiples of 32. With a 1 MP budget, one square reference
is capped at 992x992, two at 704x704 each, and five at 416x416 each.
Smaller references are not enlarged, apart from the model's 32px minimum side.
Their unused share is not redistributed. Alpha and reference order are preserved.
Use `{"reference_megapixels": 0.5}` or CLI
`--reference-megapixels 0.5` for a smaller budget. Finite values from 0.001024 to
4.3008 are accepted. The total budget must also allow at least 32x32 pixels per
reference; otherwise the request is rejected. This option applies to both Base
and Turbo editing.

Omit `width`, `height` and `output_resolution` to target `reference_megapixels`
for the output at the **original last reference's aspect ratio**, even for a
small input. Output dimensions round down to multiples of 32 without exceeding
the requested area: a 100x30 reference produces a 1824x544 result at 1 MP.
The full requested MP is used for the automatic output regardless of the number
of references: two large square inputs at 1 MP become 704x704 each, while the
output is 992x992. A small reference itself remains small.
Pass both dimensions to choose a canvas, or
`output_resolution` to choose its approximate square-root area while keeping
the last reference's aspect ratio. These overrides affect the result only.

References are also downscaled to at most 2752px per side. Encoded input limits
(25 MiB per file, 8192px per source side) still apply before resizing. The megapixel
budget limits the combined reference area; GPU memory also depends on the output
canvas, CFG, LoRAs and model variant, so requests can still cause CUDA OOM.
Output dimensions still
must be at least 256px per side. References keep their alpha; the vision encoder
composites RGBA over white, and the VAE still reads the alpha channel.

Docker applies `patches/diffusers-native-references.patch` to the pinned
Diffusers pipeline at build time. It adds independent reference sizing; the
build fails if the patch no longer applies. When running inference outside
Docker, apply the same patch from the environment's `site-packages` directory.

For both generation and editing, omit `seed` (or pass `null`) for a fresh random
32-bit seed on every request. Explicit seeds, including `0`, are preserved, and
the response reports the seed actually used. When editing a generated image,
omit its generation seed to avoid replaying the same initial noise.

There is no separate mask field. Paint the region on a reference (circle,
annotation, or mask image) and include that picture in `images`.

Transparent output is a prompt, not a flag. The model card recommends starting
with: `This is an RGBA image with transparency. … The image has alpha channel
and the background is transparent.` Results are PNG, so an RGBA image from the
pipeline keeps its alpha.

Prompt rewriting (the separate Qwen3.5-VL 9B PE checkpoints) is not part of
this worker.

### Additional LoRAs

Both generation and edit accept the Krea 2 API format:

```json
"loras": [
  {"name": "my_style", "strength": 0.8},
  {"name": "my_character", "strength": 0.5, "type": "lora"}
]
```

Put compatible **Qwen-Image-2.1 transformer LoRAs** in `LORA_DIR` as
`<name>.safetensors`. The catalog is refreshed on requests with adapters, so a living worker
can discover newly uploaded files. Names are exact filename stems, without paths, URLs or
the `.safetensors` suffix. Symlinks and obsolete Viggle accelerator files are excluded.
No adapters are downloaded from request URLs.

Each request stages only its selected nonzero-strength adapters to
`LORA_CACHE_DIR` on container disk before Diffusers loads them. Warm requests
reuse complete copies, restore missing/incomplete local files, and refresh changed
source files (size/mtime/ctime). Transfers publish atomically and never use
network-backed symlinks or hardlinks. `stage=lora_download` logs distinguish copies
and cache hits. Qwen waits for adapter loading before entering its Diffusers
pipeline. Allow container disk space for the adapters used over the worker's
lifetime; the cache has no automatic eviction and disappears with the container.

`loras` defaults to `[]`. Up to **four** unique additional adapters are allowed.
`strength` defaults to `1.0`, must be finite and in `0..2`; zero skips the
adapter after validating its name. `type` defaults to `"lora"`. Krea 2's
`"weight_diff"` format is **unsupported** here. Text encoder/VAE adapters,
missing A/B pairs and incompatible layer dimensions are rejected. Diffusers
converts supported Qwen checkpoint formats before validation; a LoRA made for
another Qwen version is not automatically compatible with 2.1.

ComfyUI Qwen 2.1 adapters with fused `img_mlp.gate_up` A/B weights are
supported: the worker splits B's `[gate; up]` output rows into Diffusers'
`gate_layer` and `proj`, sharing A. PEFT target/rank/alpha metadata is updated
for the split while preserving adapter scaling. Broken pairs, odd output-row
counts and checkpoints mixing fused and separate weights are rejected.

There is no built-in adapter. Requested adapters stay unmerged and are
loaded for each request, then removed even if generation fails. Requests on the same pipeline are serialized to
prevent adapter state from leaking between jobs. Each extra LoRA consumes
memory and adds local loading time; leave GPU/RAM headroom.

Unknown or incompatible adapters return an `error` without `refresh_worker`.
CUDA OOM retains the existing worker restart behavior.

### Response

```json
{
  "model_variant": "turbo",
  "model_id": "Qwen/Qwen-Image-2.1-Turbo",
  "sampler": "euler",
  "scheduler": "checkpoint",
  "images": ["data:image/png;base64,..."],
  "seed": 42,
  "width": 1024,
  "height": 1024,
  "type": "image_generate",
  "num_inference_steps": 8,
  "guidance_scale": 1.0,
  "output_resolution": 1024,
  "loras": [{"name": "my_style", "strength": 0.8, "type": "lora"}]
}
```

Images are returned only in the `images` array; there is no `image_url` alias.
Edits also return `num_refs`. With `BUCKET_ENDPOINT_URL` set, `images` are
URLs instead of data URLs.
`loras` reports effective additional adapters (zero-strength entries are omitted).
Jobs without requested LoRAs run directly on their selected official checkpoint.

## Local client

```bash
export RUNPOD_API_KEY=...
export ENDPOINT_ID=...
python scripts/qwen_image_21.py --prompt "a red fox" -o out.png
python scripts/qwen_image_21.py --prompt "a red fox" \
  --lora 'my_style@0.8,my_character@0.5' -o styled.png
python scripts/qwen_image_21.py -i a.png -i b.png \
  --prompt "put both subjects in one photo" -o edit.png
python scripts/qwen_image_21.py --model-variant base -i a.png \
  --prompt "change the background to a sunset beach" -o base-edit.png
python scripts/qwen_image_21.py --model-variant base --steps 40 --cfg 3.5 \
  --sampler er_sde -i a.png --prompt "make it sunset" -o custom-edit.png
```

The CLI uses the same `ENDPOINT_ID` for both variants. `--steps` may be omitted
or overridden; `--cfg` aliases `--guidance-scale`. A Base API edit can omit steps:

```json
{"input":{"type":"image_edit","model_variant":"base","prompt":"make it sunset","images":["data:image/png;base64,..."]}}
```

Or override sampling explicitly:

```json
{"input":{"type":"image_edit","model_variant":"base","prompt":"make it sunset","images":["data:image/png;base64,..."],"num_inference_steps":40,"guidance_scale":3.5,"sampler":"er_sde"}}
```

## Tests

Tests do not download model weights or require a GPU. Handler/runtime tests
replace RunPod and Diffusers dependencies with local doubles:

```bash
python -m unittest discover -s workers/qwen_image_21/tests -p 'test_*.py' -v
```

With Torch, Diffusers, PEFT and safetensors installed, the suite also checks
actual adapter composition, alpha metadata and cleanup on a tiny CPU model.
The native-resolution integration test applies the Docker patch in a temporary
directory and checks reference dimensions through the actual pipeline's text,
VAE and transformer inputs, without loading model weights.
Those integration tests are skipped when these dependencies are absent.
The CPU sampling tests compare ER-SDE step by step with a pinned ComfyUI
reference, check Heun's numerical behavior, and execute the actual Qwen denoising
loop with tiny components to verify step overrides, CFG, seeded noise and reuse.
