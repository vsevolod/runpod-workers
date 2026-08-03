# RunPod worker — lora_downloader

Thin **CPU** serverless worker: batch-download CivitAI LoRA files onto the
shared Network Volume for [krea2](../krea2/) runtime adapters.

| | |
|--|--|
| **Compute** | Serverless **CPU** (no GPU, no torch) |
| **Write path** | `$LORA_DIR` (default `/runpod-volume/krea2/loras`) |
| **Auth** | `CIVITAI_TOKEN` endpoint secret |
| **Design** | [lora_downloader design](../../docs/superpowers/specs/2026-08-03-lora-downloader-design.md) |

HTTP client code is **original** (stdlib `urllib`). This repo does **not**
vendor [civitai-downloader](https://github.com/ashleykleynhans/civitai-downloader)
(GPL-3.0).

## Layout

```
workers/lora_downloader/
├── Dockerfile
├── handler.py
├── download.py
├── schemas.py
├── requirements.txt
├── test_input.json
├── NOTICE
├── LICENSES/
└── tests/
```

## Deploy invariants (required)

1. **Serverless CPU** endpoint (not GPU).
2. **Same network volume + datacenter** as the krea2 endpoint.
3. Volume mounts at `/runpod-volume`.
4. **`workersMax = 1`** — this endpoint is the **only writer** of `$LORA_DIR`.
   Do not raise max workers (RunPod warns about concurrent volume writes).
5. Secret: `CIVITAI_TOKEN=<civitai api key>`.
6. Optional env: `LORA_DIR=/runpod-volume/krea2/loras` (must resolve **under**
   hard-coded `/runpod-volume`; must not be a symlink directory).
7. Idle timeout can be low; execution timeout must allow large multi-file batches.

### GitHub / image

- Dockerfile path: `workers/lora_downloader/Dockerfile`
- Build context: repository root

```bash
docker build -f workers/lora_downloader/Dockerfile -t runpod-lora-downloader .
```

## Network volume

```text
/runpod-volume/krea2/loras/
  my_style.safetensors   # krea2 LoRA id = my_style
```

After new files appear, **restart warm krea2 workers** so they re-scan the
allowlist (krea2 catalogs stems only at process start).

## Env

| Variable | Default | Meaning |
|----------|---------|---------|
| `CIVITAI_TOKEN` | (required) | CivitAI API key |
| `LORA_DIR` | `/runpod-volume/krea2/loras` | Sole write directory |
| `LOG_LEVEL` | `INFO` | Logging |

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

| Field | Level | Required | Default | Notes |
|-------|-------|:--------:|---------|-------|
| `items` | job | yes | — | list, length 1…20 |
| `items[].model_version_id` | item | yes | — | positive int or digit string; **not** bool |
| `items[].filename` | item | no | from CivitAI response | if key present, must be a string |
| `items[].nsfw` | item | no | `true` | **strict bool**; `true` → civitai.red |

No job-level `dest` field.

### Filename rules

Final name (override or extracted) must:

- end with `.safetensors` (case-sensitive)
- have a non-empty stem that is not `.` or `..`
- contain no `/`, `\`, or control characters
- allow Unicode and internal spaces

### Conflict policy

| Target path state | Result |
|-------------------|--------|
| missing | download |
| regular file | `skipped` / `already_exists` |
| symlink, directory, FIFO, device, etc. | `failed` |

If `filename` override is set and the target already exists as a regular file,
the worker **skips without calling CivitAI**.

### Success output (flat handler return)

RunPod wraps the handler return value as `output`. The handler returns:

```json
{
  "dest": "/runpod-volume/krea2/loras",
  "results": [
    {
      "model_version_id": "46846",
      "filename": "my_style.safetensors",
      "status": "downloaded",
      "bytes": 123,
      "path": "/runpod-volume/krea2/loras/my_style.safetensors"
    }
  ],
  "summary": { "downloaded": 1, "skipped": 0, "failed": 0 },
  "note": "Restart warm krea2 workers to pick up new LoRA files."
}
```

Per-item failures do not fail the whole job; check `summary.failed`.

### Job-level errors

Missing/invalid `input.items`, missing `CIVITAI_TOKEN`, or invalid `LORA_DIR`
return `{"error": "..."}`.

## CLI

```bash
export RUNPOD_API_KEY=...
export ENDPOINT_ID=...   # required (no hardcoded default)

python scripts/download_lora.py 46846
python scripts/download_lora.py 46846 --filename 46846=my_style.safetensors
python scripts/download_lora.py 46846 --sfw
```

- `--sfw` sets `nsfw: false` for all items
- Prints unwrapped worker `output` JSON to stdout
- Exit code **non-zero** if job error or `summary.failed > 0`

## License

See [NOTICE](NOTICE) and [LICENSES/](LICENSES/). RunPod handler patterns are
adapted under MIT. LoRA weights and CivitAI ToS are the operator's responsibility;
weights are not stored in this repository.
