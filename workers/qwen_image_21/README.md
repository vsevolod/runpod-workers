# RunPod worker — Qwen-Image-2.1

Thin serverless worker: text-to-image and multi-reference edit through
Diffusers `QwenImage21Pipeline`. No ComfyUI. Same handler shape as
[`workers/krea2`](../krea2/) (load once, base64 or S3, FlashBoot).

| | |
|--|--|
| **Model** | [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (bf16 snapshot, ~33 GB) plus the [Viggle turbo v0.2.1](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) LoRA (rank 256, ~1.4 GB) |
| **Pipeline** | `QwenImage21Pipeline` from Diffusers, pinned to commit `6256aa7` (PR [#14804](https://github.com/huggingface/diffusers/pull/14804)). The LoRA stays unmerged. The scheduler runs with `shift_terminal` unset |
| **Text encoder** | Qwen3-VL 8B, loaded with the snapshot |
| **VAE** | 64-channel RGBA VAE shipped in the same repo. The Qwen-Image / Krea 2 VAE is not interchangeable |
| **GPU** | **24 GB** with `OFFLOAD=cpu` at 1024. Native 2048 wants a larger GPU or `OFFLOAD=sequential` |
| **License** | [Qwen Research License](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/LICENSE). Research and evaluation only, unless you have a commercial agreement with Qwen |

## Layout

```
workers/qwen_image_21/
├── Dockerfile
├── handler.py
├── schemas.py
├── download_weights.py
├── test_input.json
├── requirements.txt
└── qwen_image_21_infer/
```

## Network volume

Mount the volume in the same datacenter as the endpoint.

```text
/runpod-volume/qwen_image_21/
  model_index.json
  transformer/
  text_encoder/
  vae/
  processor files…
  loras/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors
  loras/my_style.safetensors
  loras/my_character.safetensors
```

The base snapshot is about **33 GB** and the LoRA is about **1.4 GB**. Give the volume **≥ 40 GB**.

```bash
pip install huggingface-hub hf_transfer
export HF_TOKEN=hf_...    # accept the model license on the Hub first
python workers/qwen_image_21/download_weights.py \
  --output /runpod-volume/qwen_image_21
```

`download_weights.py` writes a Diffusers snapshot into `MODEL_DIR`. The worker
prefers that directory when `model_index.json` is present, otherwise it loads
`MODEL_ID` from the Hub.

## Endpoint env

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_DIR` | `/runpod-volume/qwen_image_21` | Local snapshot. Used when `model_index.json` is there |
| `MODEL_ID` | `Qwen/Qwen-Image-2.1` | Hub id when the snapshot is missing |
| `LORA_DIR` | `<MODEL_DIR>/loras` | Local directory for additional adapters. Does not change the built-in Viggle path |
| `OFFLOAD` | `cpu` | `cpu` (model offload, 24 GB), `sequential` (tighter), `none` (resident on CUDA) |
| `LOCAL_FILES_ONLY` | unset | `1` after the snapshot is warm |
| `HF_HOME` | (hf default) | Put on the volume if you load by Hub id instead of `MODEL_DIR` |
| `BUCKET_ENDPOINT_URL` | unset | If set, upload via RunPod S3 helpers instead of base64 |
| `TORCH_COMPILE_DISABLE` | `1` (Dockerfile) | Skip torch.compile |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduce allocator fragmentation |
| `QWEN_STARTUP_TRACE_INTERVAL` | `60` | Dump Python thread stacks every N seconds during model loading; `0` disables dumps |
| `QWEN_DISABLE_MMAP` | `1` | Read base weights into RAM instead of memory-mapping the network volume; `0` restores the original loading path |

`OFFLOAD=cpu` keeps one component on GPU at a time
(`text_encoder → transformer → vae`). bf16 weights do not fit together in 24 GB:
the transformer is ~14 GB and Qwen3-VL-8B is ~16 GB. The turbo LoRA stays
attached to the transformer (about 1.4 GB, not merged into the bf16 weights).

### Network-volume loading experiment

`QWEN_DISABLE_MMAP=1` is enabled by default to compare cold-start performance
against mmap on the network volume. The text encoder is loaded separately
with Transformers' `disable_mmap=True`, then passed into the pipeline.
Diffusers loads the transformer and VAE with the same option. Turbo and
request-scoped LoRA loading remain as before.

This reads base weights eagerly and increases startup RAM usage. Compare
the **whole** `stage=startup` duration, not only `stage=placement_none`:
file reading now happens inside `text_encoder_load` and
`pipeline_components_load`, before placement on GPU. Check cgroup memory
metrics as well. The runtime logs `disable_mmap=True` when enabled.

For an A/B comparison or rollback, set `QWEN_DISABLE_MMAP=0` and replace
the worker. A new image is required before this setting takes effect.

### Startup diagnostics

At `INFO` level the worker logs `stage=<name> begin/done/failed`, elapsed
seconds, PID, process RSS, and available cgroup memory usage/limit/OOM counters.
CUDA free/total/allocated/reserved memory is included after CUDA initializes.
Runtime logs include library versions, Torch CPU thread count, GPU name,
compute capability and architectures compiled into Torch.

Separate stages cover imports, source resolution, catalog scanning, base model
loading, turbo LoRA loading and activation, scheduler setup, CUDA probing,
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
| **Volume** | the snapshot above, same datacenter |
| **Container disk** | 20+ GB |

A plain `git push` may not rebuild the worker. Publish a GitHub Release so
RunPod picks up a new build.

## API

`guidance_scale` is Diffusers `true_cfg_scale`. The turbo student is sampled
**without** classifier-free guidance, so the default is `1.0` (off). A value
above 1 requires `negative_prompt` and doubles each step.

`num_inference_steps` defaults to **6**, with the v0.2.1 sigma nodes
`1, 0.9375, 0.875, 0.75, 0.5, 0.25`. Allowed counts are 4, 5, 6, 7, and 8.
4 is the training schedule and sends no custom sigmas. 5 and 7 keep the
trained low-noise nodes and only split the top of the schedule. 8 is the
dense-text schedule (`…, 0.625, 0.5, 0.25, 0.125`). Other counts are rejected:
a uniform 40-step schedule does not match this adapter.

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
    "num_inference_steps": 6,
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
    "num_inference_steps": 6
  }
}
```

Omit both `width` and `height` and the canvas follows the **last** reference
at `output_resolution` (default 1024, meaning about a 1024×1024 area). Pass
both dimensions to force a canvas. References keep their alpha; the vision
encoder composites RGBA over white, and the VAE still reads the alpha channel.

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
`<name>.safetensors`. The catalog is scanned at worker startup; restart workers
after adding files. Names are exact filename stems, without paths, URLs or
the `.safetensors` suffix. Symlinks and the built-in Viggle file are excluded.
No adapters are downloaded from request URLs.

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

Viggle turbo stays active at strength **1.0** alongside the requested LoRAs.
Additional adapters stay unmerged and are loaded for each request, then removed
even if generation fails. Requests on the same pipeline are serialized to
prevent adapter state from leaking between jobs. Each extra LoRA consumes
memory and adds local loading time; leave GPU/RAM headroom.

Unknown or incompatible adapters return an `error` without `refresh_worker`.
CUDA OOM retains the existing worker restart behavior.

### Response

```json
{
  "images": ["data:image/png;base64,..."],
  "image_url": "data:image/png;base64,...",
  "seed": 42,
  "width": 1024,
  "height": 1024,
  "type": "image_generate",
  "num_inference_steps": 6,
  "guidance_scale": 1.0,
  "output_resolution": 1024,
  "loras": [{"name": "my_style", "strength": 0.8, "type": "lora"}]
}
```

Edits also return `num_refs`. With `BUCKET_ENDPOINT_URL` set, `images` are
URLs instead of data URLs.
`loras` reports effective additional adapters (zero-strength entries are omitted).
The always-active Viggle turbo adapter is not included in that list.

## Local client

```bash
export RUNPOD_API_KEY=...
export ENDPOINT_ID=...
python scripts/qwen_image_21.py --prompt "a red fox" -o out.png
python scripts/qwen_image_21.py --prompt "a red fox" \
  --lora 'my_style@0.8,my_character@0.5' -o styled.png
python scripts/qwen_image_21.py -i a.png -i b.png \
  --prompt "put both subjects in one photo" -o edit.png
```

## Tests

Tests do not download model weights or require a GPU. Handler/runtime tests
replace RunPod and Diffusers dependencies with local doubles:

```bash
python -m unittest discover -s workers/qwen_image_21/tests -p 'test_*.py' -v
```

With Torch, Diffusers, PEFT and safetensors installed, the suite also checks
actual adapter composition, alpha metadata and cleanup on a tiny CPU model.
Those integration tests are skipped when these dependencies are absent.
