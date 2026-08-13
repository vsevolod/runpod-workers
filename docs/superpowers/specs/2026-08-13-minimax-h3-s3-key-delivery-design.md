# MiniMax H3 — S3 key delivery (Design)

**Дата:** 2026-08-13  
**Статус:** approved (brainstorm)  
**Worker:** `workers/minimax_h3_comfy/`  
**Parent:** [`2026-08-05-minimax-h3-comfyui-serverless-design.md`](./2026-08-05-minimax-h3-comfyui-serverless-design.md)

## Goal

Когда MP4 не влезает в лимит ответа RunPod (`/run` 10 MB, `/runsync` 20 MB; inline default 7e6 raw bytes), воркер заливает файл в S3-compatible storage и отдаёт **координаты объекта**, не байты и не URL.

Потребитель (другой сервис) сам делает `GetObject` своими S3-кредами.

## Non-goals

- Presigned / public `video_url`
- Примонтированный Network Volume на serverless-воркере (веса остаются на Model Cache)
- `rp_upload.upload_file_to_bucket` (после Put всегда зовёт `generate_presigned_url`; на RunPod S3 API presign **не поддерживается**)
- Endpoint / region в JSON-ответе (у потребителя уже есть креды и endpoint)
- Lifecycle / TTL / cleanup объектов
- Скачивание MP4 из CLI по S3-кредам

## Почему не RunPod-том как mount и не presign

- Mount тома лочит GPU в DC тома — от этого ушли ради Model Cache.  
- RunPod S3 API: `PutObject` / `GetObject` есть, `GeneratePresignedURL` нет.  
- Арендованный том доступен по S3 **без** аттача к endpoint: `BUCKET_NAME` = id тома, endpoint = `https://s3api-<DC>.runpod.io/`.

## Delivery matrix

| `BUCKET_*` | Поведение |
|------------|-----------|
| Все четыре непустые | Upload → `delivery: "s3"` + `bucket` + `key` |
| Все пустые | Inline `video` base64 если `size ≤ MAX_INLINE_VIDEO_BYTES` (default 7e6); иначе ошибка с текстом про `BUCKET_*` |
| Частичный набор | **Process exit на старте** (как сейчас) |

`BUCKET_REGION` опционален и **не** входит в all-or-nothing четвёрку.

## Product response (S3 mode)

```json
{
  "delivery": "s3",
  "bucket": "vol_xxxxxxxx",
  "key": "abc123/MiniMax_H3_00001_.mp4",
  "bytes": 18432000,
  "filename": "MiniMax_H3_00001_.mp4",
  "width": 864,
  "height": 480,
  "duration": 5.0,
  "seed": 42,
  "model": "minimax_h3_fl2va_pruned_int8_convrot",
  "mode": "t2v",
  "prompt_id": "…"
}
```

| Поле | Правило |
|------|---------|
| `delivery` | `"s3"` (не `"url"`) |
| `bucket` | `BUCKET_NAME` as-is |
| `key` | `{job_id}/{SaveVideo filename}` |
| `video` / `video_url` | **отсутствуют** |
| `endpoint` / `region` | **отсутствуют** |

Потребитель: `GetObject(Bucket=bucket, Key=key)` на заранее известном endpoint.

## Env

| Переменная | Обязательна для S3 | Смысл |
|------------|--------------------|--------|
| `BUCKET_ENDPOINT_URL` | да | `https://s3api-<DC>.runpod.io/` (или любой S3-compatible) |
| `BUCKET_ACCESS_KEY_ID` | да | RunPod S3 API key access (не обычный RunPod API key) |
| `BUCKET_SECRET_ACCESS_KEY` | да | secret этой пары |
| `BUCKET_NAME` | да | id тома / имя бакета |
| `BUCKET_REGION` | нет | если пусто и host = `s3api-<dc>.runpod.io` → `<DC>` upper-case (`EU-RO-1`) |

Том к serverless endpoint **не** аттачится.

## Upload

Свой boto3 `upload_file` (SigV4), `ContentType=video/mp4`.  
Не вызывать `generate_presigned_url`.  
Ошибка upload → job error (строка в `{"error": …}`), как остальные handler failures.

## CLI

`scripts/minimax_h3_t2v.py`: при `delivery == "s3"` печатает `bucket` и `key`, пишет `--save-json` если просили, **не** качает файл и не падает с «no video_url».

## Tests

- `region_from_endpoint`: RunPod host → DC; explicit `BUCKET_REGION` побеждает; чужой endpoint → `None` если env пуст.  
- `deliver_video` + full bucket: `delivery/bucket/key/bytes`, нет `video_url`.  
- `_upload_video`: key = `{job_id}/{name}`, `upload_file` вызван, presign не вызван (boto3 client mock).  
- Существующие none / oversized / partial тесты остаются зелёными.

## Files

| File | Change |
|------|--------|
| `workers/minimax_h3_comfy/handler.py` | S3 object delivery; boto3 upload; region helper |
| `workers/minimax_h3_comfy/tests/test_delivery.py` | Новые кейсы |
| `workers/minimax_h3_comfy/requirements.txt` | `boto3` |
| `workers/minimax_h3_comfy/README.md` | Delivery table + env |
| `scripts/minimax_h3_t2v.py` | S3-ref print path |

## Supersedes

Строка parent spec «All four → URL upload / `video_url`»: для этого воркера заменяется на S3 key delivery выше. Base64 / partial-exit без изменений.
