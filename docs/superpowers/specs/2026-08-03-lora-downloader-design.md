# Serverless worker: lora_downloader

**Дата:** 2026-08-03  
**Статус:** согласовано для планирования реализации (rev 3 — path/filename/CLI)

## Контекст

Monorepo `runpod-workers` содержит thin serverless workers (`krea2`, `joycaption`).
Krea 2 runtime LoRA читает заранее размещённые файлы с Network Volume:

```text
/runpod-volume/krea2/loras/*.safetensors
```

Stem файла становится публичным LoRA id. Скачивание по URL во время generate job
намеренно out of scope для `krea2` (см. runtime LoRA design).

Сейчас LoRA ставятся вручную: поднять Pod с volume, скачать с CivitAI через API
key. Это неудобно автоматизировать.

**Факт о RunPod:** Network Volume на Serverless **не read-only**. Монтируется в
`/runpod-volume`. Документация предупреждает о data corruption при параллельной
записи с нескольких workers — concurrent write **запрещён** на уровне deploy
(см. Deploy invariant ниже).

## Цели

- Serverless **CPU** worker `lora_downloader`: job → скачать одну или несколько
  CivitAI LoRA → записать в `$LORA_DIR` на Network Volume.
- Batch: список `model_version_id` + опциональный `filename` на элемент
  (max **20** items per job).
- Auth только через env `CIVITAI_TOKEN` (secret endpoint).
- Hosts: `civitai.com` (SFW) / `civitai.red` (NSFW); **nsfw default true**.
- Conflict: если целевой **обычный файл** уже есть → **skip**.
- Partial success: ошибка или невалидность **одного item** не отменяет остальные.
- Atomic write: unique partial via stdlib `tempfile` → fsync → rename.
- CLI-клиент в `scripts/`; `ENDPOINT_ID` обязателен, пока нет реального default.
- Thin image: **без** torch/CUDA; HTTP через **stdlib `urllib`** (свой код).

## Не входит в scope (v1)

- List / delete / rename LoRA на volume.
- Overwrite flag.
- Job-поле `dest` / запись вне `$LORA_DIR`.
- Произвольные download URL (только fixed CivitAI hosts + `model_version_id`).
- S3-compatible API upload path.
- Авто-restart / invalidate allowlist тёплых `krea2` workers.
- Хранение display name, trigger words, recommended strength.
- Валидация содержимого safetensors / совместимости с Krea 2.
- Межпроцессная блокировка volume (заменена deploy invariant `workersMax=1`).
- Копирование / vendoring GPL-кода civitai-downloader.

## Подход

**Thin worker + independent HTTP download (stdlib):**

- Отдельный `workers/lora_downloader/`.
- Свой небольшой download-протокол на `urllib` (redirect, disposition, stream).
  **Не** копировать и **не** vendor-ить
  [civitai-downloader](https://github.com/ashleykleynhans/civitai-downloader)
  (GPL-3.0). Протокол CivitAI публичный; реализация clean-room / original.
- Один job обрабатывает batch **последовательно**.
- **Deploy invariant:** endpoint — единственный writer на `$LORA_DIR`;
  **`workersMax = 1` обязателен** (не «рекомендация»). Без этого skip/rename
  не дают no-clobber гарантии при гонке writers.

Альтернативы отвергнуты для v1:

- Vendored/adapted GPL upstream — лицензионный конфликт с остальным monorepo.
- Shell-out к upstream CLI — GPL + слабый контроль.
- Универсальный file worker / job `dest` — scope creep / path-safety surface.
- GPU fallback — не нужен; RunPod Serverless CPU + network volume поддерживается.

## Layout

```text
workers/lora_downloader/
├── Dockerfile              # python slim / CPU; no CUDA
├── handler.py              # RunPod entry
├── schemas.py              # job-level INPUT_SCHEMA only
├── download.py             # original CivitAI HTTP + path helpers (stdlib)
├── requirements.txt        # runpod (+ minimal deps; no torch)
├── test_input.json
├── README.md               # deploy: workersMax=1, CPU, LORA_DIR
├── NOTICE                  # RunPod pattern attribution if needed
├── LICENSES/               # as needed for RunPod MIT pointers (no GPL vendor)
└── tests/
    ├── __init__.py
    └── test_download.py    # unit tests, no real network

scripts/download_lora.py    # remote operator CLI (ENDPOINT_ID required)
```

Корневой `README.md` monorepo — строка в таблице workers и scripts.

### Volume paths

| Контекст | Путь |
|----------|------|
| Volume root (code constant) | `/runpod-volume` — **не** env, hard-coded |
| LoRA dir | env `LORA_DIR`, default `/runpod-volume/krea2/loras` |

Нет job-`dest`. Нет env `VOLUME_ROOT`.

**Path safety для `LORA_DIR` (job-level, до batch):**

1. Константа `VOLUME_ROOT = Path("/runpod-volume")` в коде.
2. Прочитать env `LORA_DIR` (или default).
3. Сделать absolute path; normalize `..` (e.g. `Path(...).resolve()` with
   care — see symlink check).
4. Reject, если resolved path **не** лежит строго под `VOLUME_ROOT`
   (`Path.is_relative_to` / equivalent) — иначе job error. Равенство
   `VOLUME_ROOT` без подпути допустимо только если default layout всегда
   deeper; на практике `LORA_DIR` должен быть **внутри** volume (relative
   path non-empty under root).
5. Reject, если путь `LORA_DIR` **является symlink** (`Path.is_symlink()`
   на заданном/absolute path **до** follow, или после: не принимать
   directory-symlink как write root). Цель: нельзя увести запись через
   symlink вне ожидаемого layout.
6. Иначе `mkdir(parents=True, exist_ok=True)` для обычной директории.

Worker пишет **только** внутрь проверенного `LORA_DIR`.

```text
/runpod-volume/krea2/loras/
  my_style.safetensors      # id для krea2: my_style
  other.safetensors
```

После добавления файлов **тёплые krea2 workers** не пересканируют allowlist
до рестарта. Ответ job включает `note` с напоминанием.

## API

### Input

```json
{
  "input": {
    "items": [
      {
        "model_version_id": "46846",
        "filename": "my_style.safetensors",
        "nsfw": true
      },
      {
        "model_version_id": "99999"
      }
    ]
  }
}
```

| Поле | Уровень | Обязательно | Default | Описание |
|------|---------|-------------|---------|----------|
| `items` | job | да | — | Непустой list, **≤ 20** элементов |
| `items[].model_version_id` | item | да | — | CivitAI **model version** id (не page model id) |
| `items[].filename` | item | нет | из disposition / redirect | Итоговое имя в `$LORA_DIR` |
| `items[].nsfw` | item | нет | **`true`** | строго bool; `true` → civitai.red, `false` → civitai.com |

Поле **`dest` отсутствует** в v1.

### Validation: job-level vs item-level

`rp_validator` **не** валидирует вложенные элементы list автоматически.
Контракт:

| Уровень | Что проверяется | При нарушении |
|---------|-----------------|---------------|
| **Job** | `items` присутствует, это list, `1 ≤ len(items) ≤ 20` | Job error (`{"error": "..."}`) |
| **Job** | `CIVITAI_TOKEN` задан в env | Job error |
| **Job** | `LORA_DIR` resolves under hard-coded `/runpod-volume`; not a symlink; creatable dir | Job error |
| **Item** | `model_version_id`, `filename`, `nsfw` нормализуются **в цикле** | Item `status: failed`, batch **продолжается** |

Невалидный item (пустой id, bad filename override, wrong type) **не** валит
весь job.

### Item normalization (зафиксировано)

Каждый элемент `items[]` нормализуется **в цикле** до HTTP. Ошибка →
`status: failed` + message, следующие items продолжаются.

#### `model_version_id`

| Принято | Отклонено (item failed) |
|---------|-------------------------|
| `int` **> 0** (например `46846`) | `bool` (`True`/`False` — даже если `isinstance` int-like) |
| `str` из **только цифр**, значение **> 0** (например `"46846"`) | `0`, отрицательные, `""`, `"12a"`, float, null, list |

Нормализованная форма: **строка цифр** (для URL path), эквивалент
`str(int_value)`.

Проверка bool **до** принятия int: в Python `isinstance(True, int)` is True —
bool явно reject.

#### `nsfw`

| Принято | Default | Отклонено |
|---------|---------|-----------|
| строго `bool` (`True` / `False`) | **`True`** если ключ отсутствует | строки `"true"` / `"false"`, `0`/`1`, `"1"`, null если ключ есть с wrong type |

Отсутствие ключа → default `True`. Ключ присутствует с non-bool → item failed.

#### `filename` (optional override)

Если ключ отсутствует → имя берётся из HTTP disposition/path после redirect
chain, затем те же filename rules.  
Если ключ есть → значение должно быть `str`, пройти filename rules; иначе
item failed (до сети).

### Filename rules (override **и** upstream disposition — одинаково)

Итоговое имя (одна функция `normalize_filename`):

1. Тип: `str`. Сначала `strip()` **только** leading/trailing whitespace;
   внутренние пробелы **сохраняются**. После strip строка непустая.
2. Suffix **строго** `.safetensors` (case-sensitive; `.SafeTensors` /
   `.safetensor` → reject).
3. Stem = имя без суффикса `.safetensors`:
   - **непустой**;
   - не равен `.` и не равен `..`;
   - не содержит `/`, `\`;
   - не содержит **управляющих** символов (C0 + DEL: `ord < 32` или
     `ord == 127`; при необходимости расширить Unicode category `Cc`).
4. **Unicode и пробелы в stem допустимы** (реальные CivitAI filenames).
5. Нет автодобавления `.safetensors`: override и upstream без точного суффикса
   → failed.
6. Upstream `foo.zip` / `foo.ckpt` → failed.

Содержимое файла (safetensors magic/header) в v1 **не** проверяется.

### Auth

- Env: **только** `CIVITAI_TOKEN` (alias `CIVITAI_API_KEY` **нет**).
- Token **не** в job input, **не** в логах.
- Bearer отправляется **только** на первый HTTPS-запрос к CivitAI API host
  (`civitai.com` / `civitai.red`). На CDN / redirect URL token **не**
  пересылается (follow redirect without `Authorization`).

### Conflict / target path policy

Перед download для каждого item, после resolve filename:

| Состояние `LORA_DIR / filename` | Результат |
|----------------------------------|-----------|
| Не существует | download |
| Обычный **файл** существует | **`skipped`**, `reason: already_exists` |
| Существует **symlink** или **directory** | **`failed`** (не skip) |

Overwrite в v1 нет.

### Handler return value

RunPod сам оборачивает return value handler-а в `output`. Handler возвращает
**непосредственно** payload:

```python
return {
    "dest": str(lora_dir),
    "results": [...],
    "summary": {"downloaded": n, "skipped": n, "failed": n},
    "note": "Restart warm krea2 workers to pick up new LoRA files.",
}
```

**Не** возвращать `{"output": {...}}` — иначе двойная вложенность.

Job-level error:

```python
return {"error": "missing CIVITAI_TOKEN"}
```

### Success response shape

```json
{
  "dest": "/runpod-volume/krea2/loras",
  "results": [
    {
      "model_version_id": "46846",
      "filename": "my_style.safetensors",
      "status": "downloaded",
      "bytes": 123456789,
      "path": "/runpod-volume/krea2/loras/my_style.safetensors"
    },
    {
      "model_version_id": "99999",
      "filename": "foo.safetensors",
      "status": "skipped",
      "reason": "already_exists",
      "path": "/runpod-volume/krea2/loras/foo.safetensors"
    },
    {
      "model_version_id": "1",
      "status": "failed",
      "error": "File not found"
    }
  ],
  "summary": {
    "downloaded": 1,
    "skipped": 1,
    "failed": 1
  },
  "note": "Restart warm krea2 workers to pick up new LoRA files."
}
```

Partial item failures → всё равно successful job output с `summary.failed > 0`.

### Job-level errors

| Условие | Смысл |
|---------|--------|
| `items` missing / not a list / empty / `len > 20` | validation error |
| Нет `CIVITAI_TOKEN` | missing credentials |
| `LORA_DIR` resolved outside `/runpod-volume` | invalid LORA_DIR |
| `LORA_DIR` is symlink (or not a usable directory) | invalid LORA_DIR |
| `LORA_DIR` не может быть создан | configuration error |

## Компоненты

### `download.py` (original, stdlib)

- `normalize_filename(name) -> str` — единые правила; raises/returns error for item
- `normalize_item(raw) -> NormalizedItem | ItemError`
- `resolve_api_url(model_version_id, nsfw) -> str` — fixed hosts only
- `download_to_lora_dir(...)` — HTTP + atomic finalize
- HTTP:
  - timeouts (connect + read; конкретные константы в plan/impl)
  - first request: Bearer + User-Agent to CivitAI API
  - manual redirect loop: max **5** hops; each `Location` HTTPS only;
    **no** Authorization after first hop
  - filename from query disposition / `Content-Disposition` / URL path
  - if `Content-Length` present: verify downloaded byte count matches
  - 404 → item failed
- Write: `tempfile` unique partial **в `$LORA_DIR`** (same filesystem for
  atomic rename), stream, fsync, `os.replace` only if final still absent
  (best-effort under single-writer invariant); on failure unlink partial

### `handler.py`

- Job-level validate (`items` list length) via schema + light checks.
- Ensure `LORA_DIR` exists.
- Sequential loop: normalize item → skip/fail/download → append result.
- Return payload dict **or** `{"error": "..."}`.
- No heavy imports / no model load at startup.

### `schemas.py`

- Job-level only: `items` required list (length bounds may be enforced in
  handler if schema cannot express max 20 cleanly).
- **Не** полагаться на nested automatic validation для per-item fields.

### CLI `scripts/download_lora.py`

```bash
export RUNPOD_API_KEY=...
export ENDPOINT_ID=...          # required (no hardcoded default in v1)

python scripts/download_lora.py 46846 99999 \
  --filename 46846=my_style.safetensors

# SFW host for all items (nsfw=false)
python scripts/download_lora.py 46846 --sfw
```

- `ENDPOINT_ID` **обязателен** (env или flag); fail fast если не задан.
- Позиционные version ids; optional id→filename map.
- **`--sfw`**: для всех items выставить `nsfw: false` (default CLI без флага
  → `nsfw: true`, как worker default).
- После ответа RunPod: **распаковать** job result — взять полезную нагрузку
  из `output` (или error path), не печатать сырой envelope без разбора.
- **Exit code:**
  - `0` — job успешен и `summary.failed == 0` (skipped/downloaded only);
  - **ненулевой** — job-level `error`, transport/API failure, **или**
    `summary.failed > 0`.
- Print JSON payload (unwrapped output) на stdout; diagnostics на stderr.

## Download protocol (CivitAI)

Все redirects обрабатываются **вручную** (no automatic follow that would attach
Authorization to CDN).

1. Start URL:  
   `https://civitai.red/api/download/models/{id}` if nsfw else  
   `https://civitai.com/api/download/models/{id}`  
   Headers on **first** request only: `Authorization: Bearer {token}`,
   `User-Agent`. Timeouts (connect + read) — константы в коде.
2. Если ответ 3xx:
   - требуется заголовок `Location`;
   - resolve relative against current URL;
   - схема **только HTTPS** (reject `http://` и прочие schemes);
   - **не** слать `Authorization` на следующий hop;
   - повторять до non-3xx;
   - **максимум 5** redirect hops; превышение → item failed.
3. Non-3xx после chain: 404 → failed «File not found»; другие ошибки → failed
   с безопасным сообщением.
4. На финальном response (stream body): extract filename from disposition /
   query / path; `normalize_filename`.
5. Если regular file already at target → skip (prefer before writing body when
   filename known; if disposition only mid-flight, abort without replace).
6. Else unique partial via `tempfile` in `LORA_DIR`, stream, fsync,
   `os.replace` to final.
7. If `Content-Length` present and downloaded bytes ≠ length → failed, unlink
   partial.
8. Cleanup partial on any error.

Hosts fixed; no user base URL.

## Env

| Variable | Default | Description |
|----------|---------|-------------|
| `CIVITAI_TOKEN` | (required) | CivitAI API key only |
| `LORA_DIR` | `/runpod-volume/krea2/loras` | Write directory; must resolve under hard-coded `/runpod-volume`; no symlink dir |
| `LOG_LEVEL` | `INFO` | Logging |

Нет env для volume root (константа в коде). Нет `CIVITAI_API_KEY` alias.

## Deploy (invariants)

1. Dockerfile path: `workers/lora_downloader/Dockerfile`, context repo root.
2. **Serverless CPU** endpoint (not GPU).
3. Same network volume + datacenter as krea2.
4. Volume mount → `/runpod-volume`.
5. Secret: `CIVITAI_TOKEN=...`
6. Env: `LORA_DIR=/runpod-volume/krea2/loras` (if non-default ever needed).
7. **`workersMax = 1` — обязательный invariant.** Endpoint is the only writer
   of `$LORA_DIR`. Document in README; operator must not raise max workers.
8. Idle timeout low; execution timeout enough for batch of large files
   (console; tens of minutes if needed).
9. Sequential items inside a job; no parallel downloads in-process in v1.

## Testing

Unit (no real network):

- Job-level: empty items / >20 → error
- `LORA_DIR` outside `/runpod-volume` / symlink dir → job error
- Item: `model_version_id` accept `"46846"` / `46846`; reject `True`, `0`,
  `"12a"`, float
- Item: `nsfw` default true; accept bool only; reject `"true"` / `1`
- Filename: accept `My LoRA 日本語.safetensors`; reject `a.zip`, empty stem,
  `.safetensors`, `..safetensors` stem rules, `/`, `\`, control chars,
  wrong case suffix
- Skip only for existing regular file; symlink/dir at target → failed
- Mock HTTP: up to 5 HTTPS redirects; 6th fails; Bearer only on first hop;
  HTTP Location rejected; Content-Length mismatch → failed
- Summary counters mixed statuses
- Missing token → job error
- Handler return is flat payload (not nested `output`)
- CLI: `--sfw` sets nsfw false; exit non-zero when `summary.failed > 0`

Manual smoke:

1. Deploy CPU endpoint, volume, `workersMax=1`, token
2. `ENDPOINT_ID=... python scripts/download_lora.py <version_id> --filename ...=x.safetensors`
3. Verify file in `LORA_DIR`
4. Re-run → skipped
5. Restart krea2; generate with new LoRA stem

## Interaction with krea2

```text
lora_downloader job
  → *.safetensors in LORA_DIR
  → krea2 warm workers keep old allowlist
  → operator restarts krea2 workers
  → krea2 scans stems
  → generate: {"loras": [{"name": "<stem>", "strength": 1.0}]}
```

## License / NOTICE

- **Не** копировать GPL-3.0 civitai-downloader. Собственная реализация протокола.
- В README можно упомянуть CivitAI download API / operational similarity
  **без** включения их исходников.
- RunPod handler patterns — MIT pointers как у krea2/joycaption при необходимости.
- Веса LoRA и CivitAI ToS — ответственность оператора; файлы не в git.

## Решения (зафиксировано)

| Тема | Решение |
|------|---------|
| Compute | Serverless **CPU** worker |
| Implementation | Original stdlib HTTP; **no** GPL vendor |
| Input | Batch `items[]`, max **20** |
| dest | **нет** job field; only validated `$LORA_DIR` |
| Volume root | hard-coded `/runpod-volume` (not env) |
| `LORA_DIR` | must resolve under volume root; **no symlink** dir |
| NSFW default | `true` (civitai.red); strict bool only |
| `model_version_id` | positive int or digit str; **bool forbidden** |
| Conflict | skip regular file; fail symlink/dir at target |
| Filename | strict `.safetensors`; stem non-empty ≠ `.`/`..`; no `/` `\` controls; **Unicode + spaces OK** |
| Auth env | **only** `CIVITAI_TOKEN` |
| Bearer scope | first CivitAI request only; never CDN |
| Redirects | manual, **max 5**, each hop HTTPS only |
| Handler return | flat `{dest, results, summary, note}` |
| Validation | job: list bounds + LORA_DIR; item: loop normalize → failed |
| Concurrent writers | **`workersMax=1` mandatory** single-writer |
| Partial files | `tempfile` unique in `LORA_DIR` |
| CLI | `ENDPOINT_ID` required; `--sfw`; unwrap output; exit ≠0 if error or `failed > 0` |
| Safetensors magic | out of v1 |
| krea2 refresh | manual restart; `note` in response |

## Changelog

### rev 3

- `VOLUME_ROOT` hard-coded `/runpod-volume`; validate resolved `LORA_DIR` under it;
  reject symlink `LORA_DIR`.
- Filename rules fixed: strict suffix, stem rules, controls banned, Unicode/spaces OK.
- Item normalize: positive id, bool reject, nsfw strict bool.
- Redirects: manual, max 5, HTTPS each hop.
- CLI: `--sfw`, unwrap RunPod `output`, non-zero exit on job error or `summary.failed > 0`.

### rev 2

- GPL: no vendored upstream; original urllib implementation.
- Handler: no double `output` wrap.
- Partial success: nested item validation in loop, not job schema fail.
- `workersMax=1` deploy invariant (not soft recommendation).
- Filename must be `*.safetensors` for override and upstream names.
- Removed job `dest` and env `VOLUME_ROOT`.
- Bearer only on API request; HTTPS redirects; timeouts; Content-Length check.
- Create `LORA_DIR`; symlink/dir target → failed.
- `tempfile` partials; CPU serverless; CLI requires `ENDPOINT_ID`.
- Closed opens: only `CIVITAI_TOKEN`, max 20 items, no header validation v1.
