# Gliese Qwen3.5 9B image caption worker

Replaces JoyCaption Beta One with
[prithivMLmods/Gliese-Qwen3.5-9B-Abliterated-Caption](https://huggingface.co/prithivMLmods/Gliese-Qwen3.5-9B-Abliterated-Caption).
The `workers/joycaption` directory, GHCR image name and client name are retained
for the existing RunPod endpoint integration. There is one caption model.

## Model and runtime

- Model revision: `ead05a2ba36c2e68cc03669414210bde75988185`.
- Native BF16 weights: three shards, about **18.82 GB** total. No quantization or CPU offload.
- `Qwen3_5ForConditionalGeneration`, Transformers **5.20.1**, Accelerate **1.15.0**.
- Torch **2.8.0+cu128**, torchvision **0.23.0+cu128**, CUDA **12.8.1** with Blackwell support.
- Use a 24 GB or larger GPU. Actual GPU headroom, speed and image quality still
  require validation of this candidate image; the old JoyCaption measurements
  do not describe Gliese.

The model's own expert-caption chat template supplies the system instruction.
The user message contains an image slot and the caption instruction.
`enable_thinking=False` requests caption output without a reasoning section.

## Download and startup

The worker starts a **CPU-only subprocess** through `shared.prefetch` before
importing Torch, Transformers and Accelerate. `shared.model_download` stages
only the selected pinned metadata and Safetensors shards on **container disk**.
GGUF and alternative templates are not downloaded.

1. Processor, tokenizer, config, chat template, generation config and shard
   index are downloaded and size-validated together.
2. The processor loads while weight downloads continue. Model construction
   uses Accelerate's empty-weight context instead of allocating a CPU copy.
3. Each shard is published individually. Immediately before loading that
   shard, the worker waits for its readiness and loads it directly to GPU.
   Later shards can download concurrently.
4. The final `all` barrier verifies successful downloader exit before the
   worker begins accepting requests. A later download failure fails startup;
   a loading failure releases the partial model and terminates/reaps the child.

An atomic manifest records the immutable revision, expected sizes and completed
files. Warm startup reuses validated files without Hub access. Missing or
truncated files are repaired; retries retain the same pinned revision.
Caption inference makes no Hub requests. Completed files survive for the
lifetime of that container disk; a fresh worker may need to download them again.

Optional offline prefetch, using the same downloader:

```sh
python workers/joycaption/download_weights.py --output /models/gliese_qwen35_9b
```

## Endpoint configuration

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_DIR` | `/models/gliese_qwen35_9b` | Local pinned checkpoint and manifest |
| `HF_HOME` | `/models/hf` in the image | Local Hub transfer cache |
| `CAPTION_MAX_PIXELS` | `1048576` | Internal processor image-area cap; allowed 65536..4194304 |
| `MAX_IMAGE_PIXELS` | `25000000` | Maximum decoded source image area |
| `LOG_LEVEL` | `INFO` | Logging |

The processor resizes to an area between 65536 and `CAPTION_MAX_PIXELS` pixels.
The 1 MP default limits image-token/activation memory on 24 GB GPUs while still
accepting larger source images within the decode limit.

For deployment on the existing caption endpoint:

- Build and validate the immutable `runpod-workers-joycaption` image.
- Set `MODEL_DIR=/models/gliese_qwen35_9b` and `HF_HOME=/models/hf`; remove the
  previous JoyCaption `MODEL_ID` / `LOCAL_FILES_ONLY` environment entries.
- Allow enough container disk for weights, metadata and Hub transfer temporary
  files (at least 40 GB recommended). The previous network-volume snapshot is
  not used by this worker.
- Keep the image's `CMD`; any endpoint start-command override must be cleared.
- Update the existing endpoint only after a real GPU smoke check of the candidate.
  Source/tests and a successful image build do not prove deployed behavior.

## API

The existing client remains `python scripts/joycaption.py -i photo.jpg`.
Input remains a single base64 image or data URL:

```json
{
  "input": {
    "image": "data:image/png;base64,...",
    "prompt": "Write a detailed descriptive caption for this image.",
    "max_new_tokens": 512,
    "temperature": 0.6,
    "top_p": 0.9
  }
}
```

`prompt` is optional. The existing default is a long formal English description.
`max_new_tokens` accepts 1..1024; it is an output cap. `temperature=0` selects
non-sampling generation. The existing successful response fields are preserved:

```json
{
  "caption": "A detailed description...",
  "prompt": "Write a detailed descriptive caption for this image.",
  "model": "prithivMLmods/Gliese-Qwen3.5-9B-Abliterated-Caption",
  "model_revision": "ead05a2ba36c2e68cc03669414210bde75988185"
}
```

Errors return `error`; generation failures also request a worker refresh.
Caption duration is logged separately from processor/shard loading timings.
The client reports RunPod's `delayTime` and `executionTime` separately.

## Verification

```sh
PYTHONPATH=workers:workers/joycaption python -m pytest workers/joycaption/tests -q
```

Tests cover the caption API, image input, non-thinking template arguments,
output-token trimming, sampling controls, cache repair/retry and revision
provenance. With runtime dependencies installed, a tiny real Qwen3.5 model
loads from Safetensors shards while a controlled downloader subprocess blocks
publication of its later shard until the first shard is loaded. The same tests
cover later-download failure and child cleanup after a loading failure.
They use CPU only and do not download production weights.

Model card license: Apache-2.0. See `NOTICE` and the pinned upstream repository.
