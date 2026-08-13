# MiniMax H3 — optional S3 `file_path` (Design)

**Дата:** 2026-08-14  
**Статус:** approved (brainstorm; implement without further review)  
**Worker:** `workers/minimax_h3_comfy/`  
**Parent:** [`2026-08-13-minimax-h3-s3-key-delivery-design.md`](./2026-08-13-minimax-h3-s3-key-delivery-design.md)

## Goal

Потребитель может заранее задать object key в product input. Зная свои S3-креды и этот ключ, он делает `GetObject` сам, не дожидаясь payload джобы, чтобы узнать `{job_id}/{filename}`.

Параметр опционален. Если его нет — поведение parent spec без изменений.

## Non-goals

- Presigned / public URL
- `file_path` как `s3://bucket/key` или другой URI
- Смена бакета с запроса (ключ всегда в `BUCKET_NAME`)
- Fail-if-exists / If-None-Match
- Per-job S3 credentials
- Lifecycle / TTL / cleanup
- Скачивание MP4 из CLI по S3-кредам

## Product input

Опциональное поле рядом с `prompt` / canvas / images:

```json
{
  "input": {
    "prompt": "…",
    "file_path": "videos/abc/out.mp4"
  }
}
```

`file_path` — **только object key** (вариант A). Bucket берётся из env `BUCKET_NAME`.

## Нормализация

| Вход | Результат |
|------|-----------|
| поле отсутствует / `null` | `None` (как «не передали») |
| `""`, `"   "`, `"///"` | `None` |
| `"videos/abc/out.mp4"` | `"videos/abc/out.mp4"` |
| `"/videos/abc/out.mp4"` | `"videos/abc/out.mp4"` (снять ведущие `/`) |
| не строка | `ValueError` |
| строка с `://` (`s3://…`, `https://…`) | `ValueError` — это не key-only |

Пустой после `strip` + `lstrip("/")` = omitted, не ошибка.

## Delivery matrix

| `file_path` | `BUCKET_*` | Поведение |
|-------------|------------|-----------|
| `None` | full | как parent: `Key = {job_id}/{SaveVideo filename}` |
| задан | full | `PutObject(Bucket=BUCKET_NAME, Key=file_path)`, overwrite |
| любой | none | **игнор** `file_path`, inline base64 как parent (после очереди лучше успешный inline, чем ошибка «нет бакета») |
| любой | partial | без изменений: process exit на старте |

Ответ S3 без нового поля: `delivery` / `bucket` / `key` / `bytes`. `key` = фактически залитый ключ. `video` / `video_url` / `endpoint` / `region` отсутствуют.

Ответ base64 без `key`, даже если клиент прислал `file_path`.

## Upload

Тот же boto3 `upload_file` (SigV4), `ContentType=video/mp4`. Presign не вызывать.

`deliver_video(mp4, job_id, file_path=None)` считает ключ: `file_path or f"{job_id}/{mp4.name}"` и передаёт его в `_upload_video(path, key)`.

Ошибка upload → job `{"error": …}`, как остальные handler failures.

## CLI

`scripts/minimax_h3_t2v.py`: опциональный `--file-path`, прокидывается в product input как `file_path`. Существующий S3 print path без изменений.

## Tests

`workers/minimax_h3_comfy/tests/test_delivery.py` (CPU):

- normalize: omit / empty / slashes-only → `None`; strip leading `/`; reject non-string and URI
- `deliver_video` + S3 + `file_path` → `key` = переданный; `_upload_video` получил этот ключ
- `deliver_video` + S3 без `file_path` → `{job_id}/{name}`
- `deliver_video` + none + `file_path` → `base64`, нет `key`, upload не звался
- `_upload_video` + explicit key → `upload_file` с этим Key, presign не звался

Существующие none / oversized / partial / region тесты остаются зелёными.

## Files

| File | Change |
|------|--------|
| `workers/minimax_h3_comfy/handler.py` | parse `file_path`; pass through delivery |
| `workers/minimax_h3_comfy/tests/test_delivery.py` | новые кейсы; обновить вызов `_upload_video` если сигнатура сменится |
| `workers/minimax_h3_comfy/README.md` | поле в таблице input |
| `scripts/minimax_h3_t2v.py` | `--file-path` |

## Extends

Parent S3 key delivery: дефолтный ключ `{job_id}/{filename}` остаётся, когда `file_path` не задан. Base64 / partial-exit без изменений.
