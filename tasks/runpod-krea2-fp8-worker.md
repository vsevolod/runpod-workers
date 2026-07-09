# RunPod custom worker — Krea 2 Turbo FP8

План создания **своего** RunPod Serverless worker для генерации изображений Krea 2 (Turbo, FP8), на каркасе официального [runpod-workers/worker-sdxl](https://github.com/runpod-workers/worker-sdxl).

**Дата:** 2026-07-09  
**Статус:** план (не реализован)  
**Связано:** [2026-07-08-runpod-flux-schnell.md](./2026-07-08-runpod-flux-schnell.md) (фаза 1 ComfyUI / Schnell — отдельный MVP)

---

## Цель

| | |
|--|--|
| **Что** | Serverless endpoint: `prompt` → PNG (base64), без ComfyUI workflow JSON |
| **Модель** | Krea 2 Turbo FP8 ([AlperKTS/Krea2_FP8](https://huggingface.co/AlperKTS/Krea2_FP8)) |
| **Шаблон worker’а** | Структура и паттерны [worker-sdxl](https://github.com/runpod-workers/worker-sdxl) |
| **Инференс** | Официальный код / pipeline из [krea-ai/krea-2](https://github.com/krea-ai/krea-2) (+ при необходимости diffusers) |
| **Потребитель** | Telegram-бот: `Providers::Runpod::*` (упрощённый input, без `flux1_schnell` workflow) |
| **Позже** | Подмешивание LoRA (train на Raw, apply на Turbo) |

**Не цель этого плана:** использовать Krea cloud API (`api: krea` / `krea/krea-2-medium*`) — это уже есть в боте.

---

## Почему worker-sdxl, а не ComfyUI Hub

| | worker-sdxl (база) | worker-comfyui (Schnell) |
|--|--------------------|---------------------------|
| Runtime | Thin: Python + torch + handler | Тяжёлый ComfyUI |
| Input | Простой JSON (`prompt`, `width`…) | Полный workflow graph |
| Cold start | Меньше обвязки | Больше overhead |
| Кастомизация | Свой `handler.py` | JSON + nodes |
| LoRA позже | Параметры job + `load_lora` | Ноды в workflow |

**Важно:** из `worker-sdxl` берём **каркас** (Dockerfile, `runpod.serverless`, validate schema, base64/S3 upload, load-once), **не** `StableDiffusionXLPipeline` и не SDXL-веса.

---

## Модели (AlperKTS/Krea2_FP8)

Источник: https://huggingface.co/AlperKTS/Krea2_FP8/tree/main

| Файл | ~размер | Назначение |
|------|---------|------------|
| `krea2_turbo_fp8.safetensors` | ~12–13 GB | DiT / diffusion (Turbo, FP8) |
| `qwen3vl_4b_fp8_scaled.safetensors` | ~5.3 GB | Text encoder (Qwen3-VL 4B, FP8) |
| `qwen_image_vae.safetensors` | ~0.3 GB | VAE |
| **Итого на диске** | **~18–19 GB** | vs ~35 GB full bf16 tree |

Официальный full-precision Turbo: [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) — **не** класть в production-образ; только fallback / сравнение качества.

Рекомендованные параметры Turbo (из [krea-ai/krea-2](https://github.com/krea-ai/krea-2)):

| Параметр | Turbo default |
|----------|----------------|
| steps | `8` |
| cfg / guidance | `0` (disabled) |
| mu (timestep shift) | `1.15` |
| resolution | 1024…2048 (кратно 16) |

---

## VRAM и GPU на endpoint

| GPU VRAM | Оценка для FP8 Turbo | Рекомендация |
|----------|----------------------|--------------|
| 24 GB | С запасом, LoRA, выше res | Комфортный production |
| 16 GB | Реалистично при unload TE ↔ DiT | Хороший baseline по цене |
| 8–12 GB | Только с тяжёлым offload | Не для SLA бота |

**Диск ~18 GB ≠ VRAM ~18 GB.** При последовательной загрузке (encode → unload TE → sample DiT → VAE) пик ближе к размеру DiT + активации (~10–14 GB). Если держать всё resident — ближе к 20+ GB → безопаснее 24 GB.

**Старт:** GPU **16 GB** для smoke-test; при OOM / thrashing — **24 GB**.

---

## Архитектура

```
Telegram /image
  → UseCases::SendImagePrompt (api: runpod)
  → Providers::Runpod::SendImage  (простой input, без Comfy workflow)
  → POST https://api.runpod.ai/v2/{endpoint_id}/run
  → Worker handler.py
        load (once): TE + DiT + VAE from volume/cache
        encode prompt → sample 8 steps → decode VAE
        return base64 PNG (или S3 URL)
  → save public/uploads/… → send_photo
```

```
┌─────────────────────────────────────────────┐
│  Docker image (тонкий)                        │
│  torch, runpod, krea-2 / custom loaders       │
│  handler.py, schemas.py                       │
│  ❌ без 18 GB весов в слое (предпочтительно)  │
└─────────────────┬───────────────────────────┘
                  │ mount
┌─────────────────▼───────────────────────────┐
│  Network Volume (тот же DC, что endpoint)     │
│  /runpod-volume/krea2/                        │
│    krea2_turbo_fp8.safetensors                │
│    qwen3vl_4b_fp8_scaled.safetensors          │
│    qwen_image_vae.safetensors                 │
│    loras/   (фаза 2)                          │
└─────────────────────────────────────────────┘
```

Альтернатива: bake весов в image через `download_weights.py` (как SDXL) — проще первый деплой, тяжелее pull/registry. Для 18 GB **предпочтительнее volume**.

---

## Что взять из worker-sdxl

Структура репозитория (ориентир):

```
worker-krea2/   # или fork worker-sdxl → rename
├── Dockerfile
├── requirements.txt
├── handler.py          # ModelHandler + runpod.serverless.start
├── schemas.py          # INPUT_SCHEMA + validate
├── download_weights.py # опционально: HF → cache / volume bootstrap
├── test_input.json
├── .runpod/            # метаданные Hub/GitHub deploy (если есть)
└── README.md
```

Паттерны, которые **оставить**:

1. **Load once** при старте worker’а (`ModelHandler` / глобальный `MODELS`), не на каждый job.
2. **`runpod.serverless.utils.rp_validator.validate`** + `schemas.INPUT_SCHEMA`.
3. **Output:** base64 `data:image/png;base64,…` или `rp_upload` при `BUCKET_ENDPOINT_URL`.
4. **`rp_cleanup`** временных файлов.
5. **Dockerfile:** CUDA base → Python 3.11 → `uv`/pip → torch cu121 → `CMD python -u /handler.py`.
6. **GitHub Integration** RunPod: build image из репо.

Что **выкинуть / заменить**:

| SDXL | Krea2 |
|------|--------|
| `StableDiffusionXLPipeline` / refiner | loader TE + DiT + VAE (krea-2 / custom) |
| `num_inference_steps=25`, CFG 7.5 | steps=8, cfg=0, mu=1.15 |
| SDXL schedulers list | sampler из krea-2 (`sampling.py`) |
| `download_weights` SDXL+refiner | download 3 файла AlperKTS **или** mount volume |
| negative_prompt / high_noise_frac | не нужны в MVP (можно заглушить) |

---

## Целевой API worker’а (MVP)

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

| Поле | Тип | Default | Обязательно | Описание |
|------|-----|---------|-------------|----------|
| `prompt` | str | — | **да** | Текст (лучше EN; в боте уже есть `prepare_prompt`) |
| `width` | int | 1024 | нет | Кратно 16 |
| `height` | int | 1024 | нет | Кратно 16 |
| `seed` | int | random | нет | Воспроизводимость |
| `num_inference_steps` | int | 8 | нет | Turbo: 8 |
| `guidance_scale` | float | 0.0 | нет | Turbo: 0 |
| `mu` | float | 1.15 | нет | Timestep shift (krea-2) |

Response (как у SDXL worker):

```json
{
  "delayTime": 12000,
  "executionTime": 8000,
  "id": "…",
  "status": "COMPLETED",
  "output": {
    "images": ["data:image/png;base64,…"],
    "seed": 42
  }
}
```

Позже (не MVP):

```json
{
  "lora": "coolblue",
  "lora_scale": 0.8
}
```

---

## Пошаговый план работ

### Шаг 0 — Подготовка RunPod

- [ ] Network Volume в выбранном DC (место под ≥25 GB: модели + запас LoRA)
- [ ] Скачать на volume 3 файла из [AlperKTS/Krea2_FP8](https://huggingface.co/AlperKTS/Krea2_FP8/tree/main)
- [ ] Зафиксировать пути, например:
  - `/runpod-volume/krea2/krea2_turbo_fp8.safetensors`
  - `/runpod-volume/krea2/qwen3vl_4b_fp8_scaled.safetensors`
  - `/runpod-volume/krea2/qwen_image_vae.safetensors`
- [ ] API key уже есть (фаза Schnell)

### Шаг 1 — Репозиторий worker’а

- [ ] Fork или template: https://github.com/runpod-workers/worker-sdxl
- [ ] Переименовать в `worker-krea2` (свой org/account)
- [ ] Очистить SDXL-специфику из `handler.py` / `schemas.py` / README
- [ ] `requirements.txt`: `runpod`, `torch`, `safetensors`, `accelerate`, зависимости **krea-2** / transformers (версии сверить с [krea-ai/krea-2](https://github.com/krea-ai/krea-2) `pyproject.toml`)
- [ ] **Не** тащить xformers/refiner, если не нужны

### Шаг 2 — Инференс-ядро

Вариант A (предпочтительный): встроить логику [krea-ai/krea-2](https://github.com/krea-ai/krea-2) (`encoder.py`, `mmdit.py`, `autoencoder.py`, `sampling.py`, `inference.py`).

Вариант B: HuggingFace `diffusers` pipeline, **если** стабильно грузит FP8-файлы AlperKTS (проверить до продакшена).

- [ ] Локальный smoke на GPU (Pod или машина): prompt → PNG, 1024×1024, 8 steps
- [ ] Замерить peak VRAM (`nvidia-smi`)
- [ ] Сравнить 1–2 промпта с ComfyUI FP8 (качество)

### Шаг 3 — `handler.py` (RunPod)

- [ ] `ModelHandler.__init__` → load с volume (`MODEL_DIR` env, default `/runpod-volume/krea2`)
- [ ] `handler(job)` → validate → generate → base64 list
- [ ] Ошибки: понятный `{"error": "…"}` (нет файла, OOM, bad size)
- [ ] `runpod.serverless.start({"handler": handler})`
- [ ] Load **на init worker’а**, не lazy на первом request (важно для FlashBoot / предсказуемого cold start)

### Шаг 4 — Dockerfile

- [ ] База как у SDXL: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` (или актуальная LTS CUDA)
- [ ] **Без** `RUN python download_weights.py` на 18 GB (если volume) — только code + deps
- [ ] Если bake: multi-stage / HF token secrets, image ≥20 GB
- [ ] `CMD python -u /handler.py`

### Шаг 5 — Деплой endpoint

**Рекомендуемый путь:** **Deploy with Github repository**

- [ ] New Endpoint → GitHub → `worker-krea2`
- [ ] Attach Network Volume
- [ ] GPU: 16 GB (тест) или 24 GB
- [ ] Active workers: `0` (или `1` если нужен warm)
- [ ] Max workers: 1–3
- [ ] FlashBoot: enabled
- [ ] Container disk: 20+ GB (логи, temp; модели на volume)
- [ ] Env:
  - `MODEL_DIR=/runpod-volume/krea2`
  - при необходимости `HF_HOME`, `BUCKET_*` для S3

Альтернатива: **Deploy from a Docker image** — тот же Dockerfile, push в GHCR/Docker Hub.

**Не использовать:** Deploy LLM from Hugging Face, Hello World, Flash (для этого объёма весов), Hub Comfy (если цель — thin worker).

### Шаг 6 — Тест endpoint

```bash
export RUNPOD_API_KEY=rpa_...
export ENDPOINT_ID=...

# health
curl -sS "https://api.runpod.ai/v2/${ENDPOINT_ID}/health" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"

# run
curl -sS -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "a red apple on a wooden table, soft daylight",
      "width": 1024,
      "height": 1024,
      "seed": 42,
      "num_inference_steps": 8,
      "guidance_scale": 0.0,
      "mu": 1.15
    }
  }'
```

- [ ] Дождаться `COMPLETED`, сохранить base64 → PNG
- [ ] Записать `delayTime` / `executionTime` → оценка cost (`$/s` × execution)
- [ ] Второй запрос на warm worker — сравнить delay

### Шаг 7 — Интеграция в openai_bot

Уже есть каркас `Providers::Runpod` (фаза Schnell). Адаптация:

- [ ] Новый `endpoint_id` env: `RUNPOD_KREA2_ENDPOINT_ID` (или общий + per-model в settings)
- [ ] Ветка модели `runpod/krea-2-turbo` в `config/settings.yaml` (`api: runpod`)
- [ ] `SendImage`: **не** собирать Comfy workflow; слать `{prompt, width, height, seed, …}`
- [ ] Парсинг `output.images[0]` (data-URL / raw base64) — совместимо с SDXL-style output
- [ ] `prepare_prompt` (перевод non-EN) — как для Krea/Schnell
- [ ] `quality_costs` по фактическому `executionTime`
- [ ] Specs + `dip rspec` / rubocop

Опционально: оставить `runpod/flux-schnell` как legacy или отключить в UI.

### Шаг 8 — LoRA (после стабильного txt2img)

- [ ] Train на **Krea 2 Raw**, inference на **Turbo** (официальная рекомендация Krea)
- [ ] Класть `.safetensors` в `/runpod-volume/krea2/loras/`
- [ ] Input: `lora`, `lora_scale`
- [ ] Не держать все LoRA в VRAM — load/merge on demand или 1–2 hot

---

## Checklist файлов worker-репо (целевое состояние)

| Файл | Действие |
|------|----------|
| `handler.py` | Переписать под Krea2 |
| `schemas.py` | MVP-поля (см. API выше) |
| `Dockerfile` | Без тяжёлого bake весов (volume) |
| `requirements.txt` | krea-2 deps, без SDXL-only |
| `download_weights.py` | Либо bootstrap на volume, либо удалить |
| `test_input.json` | Krea2 prompt |
| `README.md` | Env, volume layout, example curl |
| `LICENSE` | Учесть license Krea 2 community + MIT worker template |

---

## Оценка стоимости (черновик)

Формула (как для Schnell): **cost ≈ rate_per_second × executionTime_seconds**  
(queue `delayTime` при warm обычно не биллится как GPU execution — сверять актуальные docs RunPod).

Пример: `$0.00019/s` × `12 s` ≈ `$0.0023` / image (warm).  
Cold start длиннее → выше delay для пользователя; execution всё равно ≈ sample + decode.

После замеров — прописать `quality_costs` в `settings.yaml`.

---

## Риски и решения

| Риск | Митигация |
|------|-----------|
| FP8 loader несовместим с «голым» krea-2 | Проверить AlperKTS README (Comfy-oriented); при необходимости loaders из Comfy-Org / custom quant load |
| OOM на 16 GB | Unload TE после encode; 24 GB GPU; batch=1 |
| Медленный volume | Bake hot-path весов в image **или** model caching RunPod; тот же DC |
| Cold start 1–3 мин | `min_workers=1` на пике; FlashBoot; load at init |
| Лицензия Krea 2 | [Community license](https://www.krea.ai/krea-2-licensing); commercial — opensource@krea.ai |
| Качество хуже API Krea | Сверить steps/mu/res; A/B с `krea/krea-2-medium-turbo` |

---

## Порядок «что делать руками завтра»

1. Создать Network Volume, скачать 3 файла FP8.  
2. Поднять **Pod** (не serverless) с volume + клоном `krea-ai/krea-2`, добиться 1 PNG.  
3. Fork `worker-sdxl` → вставить рабочий generate в `handler.py`.  
4. Deploy **GitHub** endpoint + volume.  
5. curl → COMPLETED.  
6. Подключить endpoint к боту (простой JSON).

---

## Ссылки

### Worker / RunPod

- [runpod-workers/worker-sdxl](https://github.com/runpod-workers/worker-sdxl)
- [runpod-workers/worker-basic](https://github.com/runpod-workers/worker-basic)
- [Deploy with GitHub](https://docs.runpod.io/serverless/workers/github-integration)
- [Handler functions](https://docs.runpod.io/serverless/workers/handler-functions)
- [Network volumes / model caching](https://docs.runpod.io/serverless/endpoints/model-caching)
- [FlashBoot / endpoint config](https://docs.runpod.io/serverless/endpoints/endpoint-configurations)

### Krea 2

- [krea-ai/krea-2](https://github.com/krea-ai/krea-2) — official inference
- [AlperKTS/Krea2_FP8](https://huggingface.co/AlperKTS/Krea2_FP8) — **production weights**
- [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) — full precision
- [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) — LoRA train
- [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) — alternative packaging / workflows
- [ComfyUI Krea 2 tutorial](https://docs.comfy.org/tutorials/image/krea/krea-2)
- [Technical report](https://www.krea.ai/blog/krea-2-technical-report)

### Бот (этот репозиторий)

- `app/providers/runpod/base.rb`
- `app/providers/runpod/send_image.rb`
- `config/settings.yaml` (`runpod/*`)
- `docs/superpowers/plans/2026-07-08-runpod-flux-schnell.md`

---

## Критерий готовности MVP

- [ ] Endpoint `COMPLETED` на curl с FP8-весами с volume  
- [ ] Peak VRAM задокументирован; выбран GPU tier  
- [ ] Бот: `/image` с моделью `runpod/krea-2-turbo` → фото в Telegram  
- [ ] `executionTime` и cost в settings согласованы  
- [ ] README worker’а: как обновить веса и задеплоить  
