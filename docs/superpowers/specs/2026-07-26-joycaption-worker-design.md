# JoyCaption RunPod serverless worker

**Дата:** 2026-07-26  
**Статус:** согласовано для планирования реализации

## Контекст

Monorepo `runpod-workers` уже содержит thin serverless worker `workers/krea2`
(image generation). Нужен второй worker: **image captioning** на
[JoyCaption Beta One](https://github.com/fpgaminer/joycaption)
([HF model](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava)).

Клиент передаёт изображение; worker возвращает English descriptive caption
(«что изображено»). Перевод и постобработка языка — **вне** этого worker
(другой шаг pipeline).

## Цели

- Serverless RunPod worker по тому же каркасу, что krea2: thin Python handler,
  веса на Network Volume, Dockerfile path `workers/joycaption/Dockerfile`.
- Input: одно изображение (base64) → output: текстовый caption.
- Default: long formal descriptive caption (native JoyCaption EN).
- Опциональный `prompt` полностью переопределяет default user-prompt.
- Load model once at process start (FlashBoot-friendly).
- bf16, GPU class **24 GB** (официально ~17 GB VRAM для модели).

## Не входит в scope (MVP)

- Video / multi-frame / temporal captioning (JoyCaption — image-only VLM).
- Image URL download; batch multi-image в одном job.
- vLLM inference stack.
- 8-bit / 4-bit quantization.
- Режимы JoyCaption через enum (`mode=straightforward|sd_prompt|...`) —
  только default descriptive + raw `prompt` override.
- Локализация caption (русский и др.) — другой шаг pipeline.
- Bake весов в Docker image.

## Подход

**Thin transformers handler** (подход 1, согласован):

- `transformers.LlavaForConditionalGeneration` + `AutoProcessor`
- `torch_dtype=bfloat16`, `device_map=0`
- Inference path **строго** как в официальном README JoyCaption
  (`apply_chat_template` + `processor(text=..., images=...)` + `generate`)
- Без ComfyUI, без vLLM

## Layout

```text
workers/joycaption/
├── Dockerfile
├── handler.py
├── schemas.py
├── download_weights.py
├── requirements.txt
├── test_input.json
└── README.md
```

Корневой `README.md` monorepo — добавить строку в таблицу workers.

## Ресурсы

| Ресурс | Значение |
|--------|----------|
| GPU | **24 GB** (bf16 ~17 GB model + generate headroom) |
| Model on disk | **~16 GB** (HF `usedStorage` ≈ 15.8 GB, 8B BF16, 4 shards) |
| Network volume | **≥ 25 GB** recommended (30–40 GB with headroom) |
| Separate volume from krea2 | **yes** (user creates new volume) |

## Volume layout

```text
/runpod-volume/joycaption/     # HF snapshot of the model
  config.json
  model-00001-of-00004.safetensors
  model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors
  model-00004-of-00004.safetensors
  model.safetensors.index.json
  tokenizer.json
  tokenizer_config.json
  preprocessor_config.json
  processor_config.json
  ...
```

Bootstrap on a **Pod** (not serverless job):

```bash
python download_weights.py --output /runpod-volume/joycaption
```

Uses `huggingface_hub.snapshot_download` for
`fancyfeast/llama-joycaption-beta-one-hf-llava`.

Worker loads from local `MODEL_DIR` with `LOCAL_FILES_ONLY=1` after volume is warm
so cold starts do not hit the network.

## Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_DIR` | `/runpod-volume/joycaption` | Local model snapshot path |
| `MODEL_ID` | `fancyfeast/llama-joycaption-beta-one-hf-llava` | HF id for download script / metadata |
| `HF_HOME` | unset | Optional HF cache root during bootstrap |
| `LOCAL_FILES_ONLY` | unset (recommend `1` after warm volume) | Offline load from `MODEL_DIR` |
| `LOG_LEVEL` | `INFO` | Logging level |

## API

### Input (`job.input`)

| Field | Type | Required | Default | Notes |
|-------|------|:--------:|---------|-------|
| `image` | `str` | yes | — | Raw base64 **or** `data:image/...;base64,...` |
| `prompt` | `str` | no | see below | Full user prompt; replaces default entirely |
| `max_new_tokens` | `int` | no | `512` | Constraints: 1…1024 |
| `temperature` | `float` | no | `0.6` | Official JoyCaption sample default |
| `top_p` | `float` | no | `0.9` | Official JoyCaption sample default |

**Default user prompt** (when `prompt` omitted):

```text
Write a long descriptive caption for this image in a formal tone.
```

**Fixed system message** (not configurable via input):

```text
You are a helpful image captioner.
```

One image per job. No URL, no video, no multi-image batch.

### Success output

```json
{
  "caption": "A golden retriever walks on wet asphalt...",
  "prompt": "Write a long descriptive caption for this image in a formal tone.",
  "model": "fancyfeast/llama-joycaption-beta-one-hf-llava"
}
```

- `caption` — primary payload (main response length).
- `prompt` — prompt actually sent to the model (default or override); useful for pipeline debugging.
- `model` — model id string for provenance.

### Error output

```json
{ "error": "human-readable message" }
```

No stack traces in the client response; full traceback only in worker logs.

## Inference flow

1. Validate `job.input` via `INPUT_SCHEMA` (runpod validator pattern as krea2).
2. Decode base64 → RGB `PIL.Image` (reject invalid data).
3. Resolve prompt: `prompt` if non-empty string provided, else default descriptive.
4. Build conversation: system + user content = resolved prompt.
5. `processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)`.
6. `processor(text=[convo_string], images=[image], return_tensors="pt").to("cuda")`.
7. Cast `pixel_values` to `bfloat16`.
8. `llava_model.generate(..., max_new_tokens, do_sample=True, temperature, top_p, use_cache=True, suppress_tokens=None)`.
9. Strip prompt tokens from `generate_ids`; decode; `strip()` → `caption`.
10. Return `{caption, prompt, model}`.

Use `@torch.inference_mode()` (or `torch.no_grad()`) around generate.

**Important:** follow the exact HF Llava chat + processor combination from JoyCaption
docs; alternate combinations can inject duplicate BOS tokens and degrade quality.

## Model loading

```text
ModelHandler at import / process start
  → AutoProcessor.from_pretrained(MODEL_DIR or MODEL_ID)
  → LlavaForConditionalGeneration.from_pretrained(..., torch_dtype=bfloat16, device_map=0)
  → eval()
```

Missing / incomplete `MODEL_DIR` fails worker startup (fail-fast), not mid-job.

## Dockerfile

- Base: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` (align with krea2).
- Python 3.12 + uv venv.
- Torch CUDA wheel (cu124) installed separately; other deps from `requirements.txt`.
- Copy only joycaption worker files into `/app`.
- No weight download at build time.
- `CMD ["python", "-u", "/app/handler.py"]`.
- RunPod: Dockerfile path `workers/joycaption/Dockerfile`, build context **repo root**.

## Dependencies (runtime)

- `runpod`
- `transformers` (version compatible with `LlavaForConditionalGeneration` + this model)
- `accelerate`
- `safetensors`
- `pillow`
- `huggingface-hub` (+ `hf_transfer` for bootstrap)
- torch/torchvision via Dockerfile CUDA index

Pin transformers in line with known-good JoyCaption usage when implementing;
verify against model card if needed.

## Error handling

| Case | Response |
|------|----------|
| Missing `input` / schema errors | `{ "error": "..." }` |
| Empty/invalid base64 / not an image | `{ "error": "invalid image: ..." }` |
| CUDA OOM / generate failure | `{ "error": "..." }`; traceback in logs only |
| Empty model dir at start | Worker process fails to start |

## Testing

| Layer | Coverage |
|-------|----------|
| Unit (no GPU) | base64 decode (raw + data URL); prompt resolve default vs override; schema constraints |
| Smoke | `test_input.json` with small base64 image on RunPod / local GPU |
| CI | Unit tests only; full generate not required without GPU |

## Deploy checklist

1. Create Network Volume ≥ 25 GB in the same datacenter as the endpoint.
2. Attach volume to a Pod; run `download_weights.py --output /runpod-volume/joycaption`.
3. Serverless → Import Git → Dockerfile `workers/joycaption/Dockerfile`, context repo root.
4. Mount volume at `/runpod-volume`.
5. GPU: 24 GB class.
6. Env: `MODEL_DIR=/runpod-volume/joycaption`, `LOCAL_FILES_ONLY=1` after warm volume.
7. Smoke job with `test_input.json`.

## Decisions log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Caption style default | Formal long descriptive EN | Native JoyCaption; language handled elsewhere |
| Image transport | base64 only | Client preference; simple serverless I/O |
| Inference stack | transformers bf16 | Official path; predictable ~17 GB; matches thin-worker style |
| Extra response fields | `prompt` + `model` | Pipeline debugging; negligible size vs caption |
| Video | Not supported | Model is image-only VLM |
| Volume | Separate from krea2 | ~16 GB model + headroom; keep deploy independent |
