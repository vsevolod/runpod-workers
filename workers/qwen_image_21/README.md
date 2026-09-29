# RunPod worker — Qwen-Image-2.1

Thin serverless worker: text-to-image and multi-reference edit through
Diffusers `QwenImage21Pipeline`. No ComfyUI. Same handler shape as
[`workers/krea2`](../krea2/) (load once, base64 or S3, FlashBoot).

| | |
|--|--|
| **Model** | [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (bf16 snapshot, ~33 GB) |
| **Pipeline** | `QwenImage21Pipeline` from Diffusers, pinned to commit `6256aa7` (PR [#14804](https://github.com/huggingface/diffusers/pull/14804)) |
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
```

The snapshot is about **33 GB**. Give the volume **≥ 40 GB**.

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
| `OFFLOAD` | `cpu` | `cpu` (model offload, 24 GB), `sequential` (tighter), `none` (resident on CUDA) |
| `LOCAL_FILES_ONLY` | unset | `1` after the snapshot is warm |
| `HF_HOME` | (hf default) | Put on the volume if you load by Hub id instead of `MODEL_DIR` |
| `BUCKET_ENDPOINT_URL` | unset | If set, upload via RunPod S3 helpers instead of base64 |
| `TORCH_COMPILE_DISABLE` | `1` (Dockerfile) | Skip torch.compile |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduce allocator fragmentation |

`OFFLOAD=cpu` keeps one component on GPU at a time
(`text_encoder → transformer → vae`). bf16 weights do not fit together in 24 GB:
the transformer is ~14 GB and Qwen3-VL-8B is ~16 GB.

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

`guidance_scale` is Diffusers `true_cfg_scale`. Qwen samples this model
**without** classifier-free guidance, so the default is `1.0` (off). A value
above 1 requires `negative_prompt` and doubles each step.

`num_inference_steps` defaults to **40**.

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
    "num_inference_steps": 40,
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
    "num_inference_steps": 40
  }
}
```

Edits return one image; `num_images` must be `1`.

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

### Response

```json
{
  "images": ["data:image/png;base64,..."],
  "image_url": "data:image/png;base64,...",
  "seed": 42,
  "width": 1024,
  "height": 1024,
  "type": "image_generate",
  "num_inference_steps": 40,
  "guidance_scale": 1.0,
  "output_resolution": 1024
}
```

Edits also return `num_refs`. With `BUCKET_ENDPOINT_URL` set, `images` are
URLs instead of data URLs.

## Local client

```bash
export RUNPOD_API_KEY=...
export ENDPOINT_ID=...
python scripts/qwen_image_21.py --prompt "a red fox" -o out.png
python scripts/qwen_image_21.py -i a.png -i b.png \
  --prompt "put both subjects in one photo" -o edit.png
```

## Tests

Contract tests do not load weights or import `handler.py`:

```bash
python -m unittest discover -s workers/qwen_image_21/tests -p 'test_*.py' -v
```
