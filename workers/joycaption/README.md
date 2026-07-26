# RunPod worker — JoyCaption Beta One

Thin serverless worker: **image → English descriptive caption**. No ComfyUI, no vLLM.

| | |
|--|--|
| **Model** | [fancyfeast/llama-joycaption-beta-one-hf-llava](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava) |
| **Upstream** | [fpgaminer/joycaption](https://github.com/fpgaminer/joycaption) |
| **GPU** | **24 GB** class (bf16 model ≈ **17 GB** VRAM) |
| **Dtype** | `bfloat16` (native) |
| **Template patterns** | [worker-sdxl](https://github.com/runpod-workers/worker-sdxl), monorepo `workers/krea2` |

## Layout

```
workers/joycaption/
├── Dockerfile
├── handler.py
├── caption_utils.py        # base64 / prompt / pixel limits (unit-tested)
├── schemas.py
├── download_weights.py
├── test_input.json
├── requirements.txt
├── NOTICE
├── LICENSES/
└── tests/
```

## Network volume

Use a **separate** volume from krea2 (same datacenter as the endpoint).

| Component | ~size |
|-----------|------:|
| HF snapshot (4× safetensors + tokenizer/processor) | **~16 GB** |
| Recommended volume | **≥ 25 GB** (30–40 GB with headroom) |

```text
/runpod-volume/joycaption/
  config.json
  model-00001-of-00004.safetensors
  ...
  tokenizer.json
  preprocessor_config.json
```

Bootstrap on a **Pod** with the volume attached:

```bash
pip install huggingface-hub hf_transfer
export HF_TOKEN=hf_...   # if needed
python download_weights.py --output /runpod-volume/joycaption
```

Then set `LOCAL_FILES_ONLY=1` on the endpoint. The worker **explicitly** passes
`local_files_only=True` into both `from_pretrained` calls (Transformers does
**not** honor that env by itself).

## Endpoint env

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_DIR` | `/runpod-volume/joycaption` | Local snapshot path |
| `MODEL_ID` | `fancyfeast/llama-joycaption-beta-one-hf-llava` | Provenance + fallback id |
| `LOCAL_FILES_ONLY` | unset | `1`/`true`/`yes` → offline load kwargs |
| `MAX_IMAGE_PIXELS` | `25000000` | Max `width * height` after decode |
| `LOG_LEVEL` | `INFO` | Logging |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` (Dockerfile) | Allocator |

## API

### Input

| Field | Required | Default | Notes |
|-------|:--------:|---------|-------|
| `image` | yes | — | Raw base64 or `data:image/...;base64,...` |
| `prompt` | no | formal long descriptive (EN) | Full override if non-empty |
| `max_new_tokens` | no | `512` | 1…1024 |
| `temperature` | no | `0.6` | JoyCaption sample default |
| `top_p` | no | `0.9` | JoyCaption sample default |

`top_k` is always `None` in `generate()` (official sample); not exposed in API.

Default user prompt:

```text
Write a long descriptive caption for this image in a formal tone.
```

### Success

```json
{
  "caption": "...",
  "prompt": "Write a long descriptive caption for this image in a formal tone.",
  "model": "fancyfeast/llama-joycaption-beta-one-hf-llava"
}
```

### Error

```json
{ "error": "human-readable message" }
```

### Payload size (RunPod)

Base64 inflates size by ~4/3. Platform limits:

| Endpoint | Payload limit | Practical raw image (approx.) |
|----------|---------------|-------------------------------|
| `/run` (async) | **10 MB** | **~7.5 MB** before base64 |
| `/runsync` | **20 MB** | **~15 MB** before base64 |

Stay under these limits on the client. After decode, `MAX_IMAGE_PIXELS` still
rejects huge decompressed dimensions (decompression bombs).

Video is **not** supported (image-only VLM).

## Deploy (RunPod)

1. Network Volume **≥ 25 GB** in the endpoint datacenter.
2. Pod + volume → `download_weights.py --output /runpod-volume/joycaption`.
3. Serverless → Import Git → Dockerfile path `workers/joycaption/Dockerfile`,
   build context **repository root**.
4. Mount volume at `/runpod-volume`.
5. GPU: **24 GB**.
6. Env: `MODEL_DIR=/runpod-volume/joycaption`, `LOCAL_FILES_ONLY=1`.
7. Smoke with `test_input.json` (tiny PNG; well under 10 MB payload).

## Local unit tests (no GPU)

```bash
cd workers/joycaption
pip install pillow pytest
PYTHONPATH=. pytest tests/ -v
```

## Design notes

- **VRAM:** official JoyCaption bf16 ≈ 17 GB; comfortable on 24 GB+.
- **Inference:** `LlavaForConditionalGeneration` + official chat/processor path.
- **Language:** English captions by design; translate in another pipeline step.
- **Not in MVP:** vLLM, quantization, URL input, multi-image, video, mode enum.

## License

- Third-party notices: [`NOTICE`](NOTICE).
- Portions adapted from RunPod `worker-sdxl`: [MIT](LICENSES/RUNPOD-WORKER-SDXL-MIT.txt).
- Model weights are not included. Llama 3.1 / SigLIP2 / JoyCaption weights are
  governed by upstream licenses (see HF model card; snapshot may include
  `LLAMA_LICENSE` and `LLAMA_USE_POLICY.md`).
