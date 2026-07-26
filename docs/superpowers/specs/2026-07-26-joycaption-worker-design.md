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
├── NOTICE                    # third-party attribution (как krea2)
├── LICENSES/                 # релевантные тексты лицензий / указатели
└── README.md                 # deploy + License section
```

Корневой `README.md` monorepo — добавить строку в таблицу workers.

### License / NOTICE (консистентность с krea2)

- В `workers/joycaption/README.md` — секция **License**, по образцу
  `workers/krea2/README.md`.
- `NOTICE` + `LICENSES/` в layout worker-а (не bake весов).
- Snapshot модели на volume содержит `LLAMA_LICENSE` и `LLAMA_USE_POLICY.md`
  (Llama 3.1). База vision: SigLIP2. README/NOTICE должны явно сказать:
  - веса **не** в репозитории;
  - использование весов регулируется лицензиями upstream (Llama + компоненты
    модели / HF card);
  - inference-код worker-а и любые адаптированные куски (например RunPod
    patterns) — с указанием лицензий в `NOTICE` / `LICENSES/`.

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

После прогрева volume выставляют `LOCAL_FILES_ONLY=1`. **Важно:** Transformers
**не** читает эту env сам. Worker должен **явно** распарсить env и передать
`local_files_only=True` в:

- `AutoProcessor.from_pretrained(...)`
- `LlavaForConditionalGeneration.from_pretrained(...)`

Паттерн как в krea2 (`workers/krea2/krea2_infer/pipeline.py`):

```python
local_files_only = os.environ.get("LOCAL_FILES_ONLY", "").lower() in {
    "1", "true", "yes",
}
```

Без этого флага `from_pretrained(MODEL_DIR)` всё ещё может ходить в сеть
(ревизии, недостающие файлы).

## Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_DIR` | `/runpod-volume/joycaption` | Local model snapshot path |
| `MODEL_ID` | `fancyfeast/llama-joycaption-beta-one-hf-llava` | HF id for download script / metadata |
| `HF_HOME` | unset | Optional HF cache root during bootstrap |
| `LOCAL_FILES_ONLY` | unset (recommend `1` after warm volume) | Parsed by worker; passed as `local_files_only=` to both `from_pretrained` calls |
| `MAX_IMAGE_PIXELS` | `25_000_000` (≈ 5000×5000) | Max `width * height` after decode; reject larger images |
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

`top_k` **не** выставляется из input: в `generate()` всегда передаётся
`top_k=None`, как в официальном примере JoyCaption (вместе с
`temperature=0.6`, `top_p=0.9`).

**Default user prompt** (when `prompt` omitted):

```text
Write a long descriptive caption for this image in a formal tone.
```

**Fixed system message** (not configurable via input):

```text
You are a helpful image captioner.
```

One image per job. No URL, no video, no multi-image batch.

### Payload size (RunPod + base64)

RunPod job payload limits (platform; document in worker README):

| Endpoint | Payload limit | Practical raw image (≈ base64 −33%, minus JSON) |
|----------|---------------|--------------------------------------------------|
| `/run` (async) | **10 MB** | ~**7.5 MB** image file before base64 |
| `/runsync` | **20 MB** | ~**15 MB** image file before base64 |

Base64 expands size by ~4/3. Clients must stay under these limits; oversized
jobs fail at the platform/API layer before the handler. README and smoke docs
must warn about this. This does **not** replace post-decode pixel limits
(decompression bomb risk).

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
3. **Pixel guard:** if `width * height > MAX_IMAGE_PIXELS` (env, default
   `25_000_000`), return
   `{ "error": "image too large: WxH exceeds MAX_IMAGE_PIXELS" }`
   (or equivalent clear message). Optional: set `Image.MAX_IMAGE_PIXELS` /
   catch `DecompressionBombError` consistently.
4. Resolve prompt: `prompt` if non-empty string provided, else default descriptive.
5. Build conversation: system + user content = resolved prompt.
6. `processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)`.
7. `processor(text=[convo_string], images=[image], return_tensors="pt").to("cuda")`.
8. Cast `pixel_values` to `bfloat16`.
9. `llava_model.generate` with **official JoyCaption kwargs**:
   - `max_new_tokens` (from input / default 512)
   - `do_sample=True`
   - `suppress_tokens=None`
   - `use_cache=True`
   - `temperature` (from input / default 0.6)
   - `top_k=None` (**always**; not exposed in API)
   - `top_p` (from input / default 0.9)
10. Strip prompt tokens from `generate_ids`; decode; `strip()` → `caption`.
11. Return `{caption, prompt, model}`.

Use `@torch.inference_mode()` (or `torch.no_grad()`) around generate.

**Important:** follow the exact HF Llava chat + processor combination from JoyCaption
docs; alternate combinations can inject duplicate BOS tokens and degrade quality.
`top_k=None` is part of that official sample and must not be omitted (HF default
`top_k` is not `None`).

## Model loading

```text
ModelHandler at import / process start
  → parse LOCAL_FILES_ONLY env → local_files_only: bool
  → AutoProcessor.from_pretrained(MODEL_DIR or MODEL_ID, local_files_only=...)
  → LlavaForConditionalGeneration.from_pretrained(
        ..., torch_dtype=bfloat16, device_map=0, local_files_only=...
    )
  → eval()
```

Missing / incomplete `MODEL_DIR` fails worker startup (fail-fast), not mid-job.
`local_files_only` must be an explicit kwarg (env alone is not enough).

## Dockerfile

- Base: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` (align with krea2).
- Python 3.12 + uv venv.
- Torch CUDA wheel (cu124) installed separately; other deps from `requirements.txt`.
- Copy only joycaption worker files into `/app`.
- No weight download at build time.
- `CMD ["python", "-u", "/app/handler.py"]`.
- RunPod: Dockerfile path `workers/joycaption/Dockerfile`, build context **repo root**.

## Dependencies (runtime)

Pin a **known-working set** (align with official JoyCaption `requirements.txt`
where practical; torch/vision still installed from CUDA wheel index in Dockerfile):

| Package | Pin guidance |
|---------|----------------|
| `torch` | Official JoyCaption cites `torch==2.7.0`; Dockerfile uses cu124 wheel — pin compatible CUDA build of that major/minor when resolving |
| `torchvision` | Compatible with chosen torch |
| `transformers` | Official: `transformers==4.51.3` (verify Llava path still works; do not leave unpinned) |
| `accelerate` | Official: `accelerate==1.6.0` |
| `pillow` | Official: `pillow==11.2.1` |
| `runpod` | `>=1.7.9` (or current monorepo floor) |
| `safetensors` | pin or floor |
| `huggingface-hub` | for bootstrap / optional load |
| `hf_transfer` | bootstrap speed |

Do **not** pin only `transformers`. Minimum explicit pins for serverless image:
**torch, torchvision, transformers, accelerate, pillow**. If cu124 wheel for
exactly 2.7.0 is unavailable at implement time, document the resolved versions
in `workers/joycaption/README.md` and keep them locked in Dockerfile /
`requirements.txt`.

## Error handling

| Case | Response |
|------|----------|
| Missing `input` / schema errors | `{ "error": "..." }` |
| Empty/invalid base64 / not an image | `{ "error": "invalid image: ..." }` |
| Image exceeds `MAX_IMAGE_PIXELS` | `{ "error": "image too large: ..." }` |
| Job payload over RunPod limit | Platform error (before handler); document client-side |
| CUDA OOM / generate failure | `{ "error": "..." }`; traceback in logs only |
| Empty model dir at start | Worker process fails to start |
| `LOCAL_FILES_ONLY` but incomplete snapshot | Fail at load with clear missing-file error |

## Testing

| Layer | Coverage |
|-------|----------|
| Unit (no GPU) | base64 decode (raw + data URL); prompt resolve default vs override; schema constraints; pixel limit reject; `local_files_only` env parsing |
| Smoke | `test_input.json` with **small** base64 image (well under 10 MB payload) on RunPod / local GPU |
| CI | Unit tests only; full generate not required without GPU |
| Docs | README states RunPod payload limits and ~7.5 MB practical async image size |

## Deploy checklist

1. Create Network Volume ≥ 25 GB in the same datacenter as the endpoint.
2. Attach volume to a Pod; run `download_weights.py --output /runpod-volume/joycaption`.
3. Serverless → Import Git → Dockerfile `workers/joycaption/Dockerfile`, context repo root.
4. Mount volume at `/runpod-volume`.
5. GPU: 24 GB class.
6. Env: `MODEL_DIR=/runpod-volume/joycaption`, `LOCAL_FILES_ONLY=1` after warm volume
   (worker must pass `local_files_only=True` into both `from_pretrained` calls).
7. Smoke job with small `test_input.json` (payload well under 10 MB).

## Decisions log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Caption style default | Formal long descriptive EN | Native JoyCaption; language handled elsewhere |
| Image transport | base64 only | Client preference; simple serverless I/O |
| Payload limits | Document RunPod 10/20 MB; warn ~7.5 MB async raw image | Base64 + platform caps; client responsibility |
| Pixel guard | `MAX_IMAGE_PIXELS` default 25M | Decompression bombs / RAM after decode |
| Offline load | Explicit `local_files_only=` from `LOCAL_FILES_ONLY` env | Transformers does not honor env automatically (krea2 pattern) |
| `generate` kwargs | Include `top_k=None` always | Official JoyCaption sample; HF default differs |
| Inference stack | transformers bf16 | Official path; predictable ~17 GB; matches thin-worker style |
| Dependency pins | torch, torchvision, transformers, accelerate, pillow | JoyCaption known-working set; stable serverless image |
| License files | NOTICE + LICENSES + README License | Parity with krea2; Llama/SigLIP upstream |
| Extra response fields | `prompt` + `model` | Pipeline debugging; negligible size vs caption |
| Video | Not supported | Model is image-only VLM |
| Volume | Separate from krea2 | ~16 GB model + headroom; keep deploy independent |
