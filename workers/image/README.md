# Common image worker

One RunPod image serves Krea 2, Qwen Image 2.1, MOSS-Image-SFT and Fibo-2.
The model registry owns routing; each backend owns its existing inference.
Requests are serialized so the old backend is closed before another loads.
The worker does not download a checkpoint until a request selects a model.

All image models use the shared CPU-only downloader and per-component readiness
protocol in `shared.prefetch`. A ready component starts loading while subsequent
components continue downloading. Krea consumes DiT -> VAE -> text encoder;
Qwen consumes metadata -> transformer -> text encoder -> VAE; MOSS consumes
transformer -> text encoder/tokenizer -> VAE/scheduler -> requested Turbo LoRA.
Fibo consumes transformer -> text encoder/processor/fonts -> VAE/scheduler.
The final `all` barrier verifies downloader success before inference. See the
repository [agent rules](../../AGENTS.md) for this required loading contract.

## API

Endpoint: `9zb0wyo61ck3wk` (`image-workers`).
The full request and response contract is in the [shared API](../shared/api/README.md).

```json
{
  "input": {
    "model": "qwen_image_21",
    "type": "image_generate",
    "prompt": "A red fox in a forest",
    "model_options": {"variant": "turbo"},
    "seed": 42
  }
}
```

`model` accepts `krea2`, `qwen_image_21`, `moss_image` and `fibo_2`. Omission selects Krea for existing
clients. The response includes `model`. Qwen Base uses `variant: "base"`.
Each model retains its own options, reference limits and compatible LoRAs.
Additional adapters remain under separate model directories on the volume.

All four bundled clients select the model explicitly and use the same endpoint:

```bash
python scripts/krea2_image.py --prompt "a red fox" --lora '' -o krea.png
python scripts/qwen_image_21.py --prompt "a red fox" --variant base -o qwen.png
python scripts/moss_image.py --prompt "a red fox" --variant turbo -o moss.png
python scripts/fibo_2.py --prompt-file scene.json -o fibo.png
```

## Configuration

| Variable | Image default | Purpose |
|---|---|---|
| `KREA_MODEL_DIR` | `/models/krea2` | Krea checkpoint and built-in adapters |
| `KREA_DIT_QUANT` | `int8_convrot` | Krea DiT quantization; also supports `fp8` |
| `KREA_LORA_DIR` | `/runpod-volume/krea2/loras` | Additional Krea LoRAs |
| `QWEN_MODEL_DIR` | `/models/qwen_image_21_turbo` | Qwen Turbo checkpoint |
| `QWEN_BASE_MODEL_DIR` | `/models/qwen_image_21_base` | Qwen Base checkpoint |
| `QWEN_LORA_DIR` | `/runpod-volume/qwen_image_21_turbo/loras` | Qwen LoRAs |
| `QWEN_MODEL_ID`, `QWEN_BASE_MODEL_ID` | Official Qwen model IDs | Optional checkpoint sources |
| `QWEN_VAE_DIR` | `/models/qwen_image_21_shared/texture_fix_vae` | Shared Qwen VAE files |
| `QWEN_VAE` | `texture-fix` | Qwen VAE profile |
| `QWEN_DISABLE_MMAP` | `0` | Keep memory mapping enabled |
| `OFFLOAD` | `cpu` | Qwen component offload for 24 GB GPUs |
| `MOSS_MODEL_DIR` | `/models/moss_image_sft` | MOSS SFT components and lazy Turbo adapter |
| `MOSS_OFFLOAD` | `cpu` | MOSS component offload; also supports `none` |
| `FIBO_MODEL_DIR` | `/models/fibo_2` | Separate Base/Turbo snapshots |
| `FIBO_OFFLOAD` | `cpu` | Fibo component offload; also supports `none` |
| `HF_HOME` | `/models/hf` | Shared local HF cache |
| `LORA_CACHE_DIR` | `/models/lora_cache` | Local staging of selected volume adapters |
| `HF_TOKEN` | Endpoint secret | HF access where required |

The model-prefixed paths override generic legacy `MODEL_DIR`, `BASE_MODEL_DIR`
and `LORA_DIR`. The local adapter cache separates source files by their
absolute paths and tracks their size/timestamps. Checkpoints remain on local
disk across model switches within the container lifetime.

Krea switches release its DiT, encoder, VAE and adapter manager. Leaving Qwen
also closes its Base/Turbo manager and retained VAE. Within Qwen, the existing
variant manager reuses the shared VAE and repairs incomplete snapshots.
MOSS variants share one SFT pipeline; Turbo adds an unmerged DMD LoRA only for
the current request. MOSS currently supports generation only and rejects user
adapters. See [MOSS settings and pinned sources](../moss_image/README.md).

Fibo accepts a VGL JSON object in `prompt`. Base supports generation/editing;
Turbo supports editing only. Both reject additional adapters. Its separate
checkpoints, pinned loading and JSON examples are documented in
[Fibo-2](../fibo_2/README.md). Fibo requires authorized gated model access and
has not been calibrated on the existing GPU profile.

## Deployment

- Dockerfile: `workers/image/Dockerfile`; context: repository root.
- Image: `ghcr.io/vsevolod/runpod-workers-image:<commit-sha>`.
- Use an immutable digest after CI passes.
- Container disk: existing 200 GB profile; allow roughly 31 GB extra for MOSS and 51 GB for both Fibo variants.
  Check actual free space when retaining all model variants.
- GPU: existing 24 GB Ampere/Ada profile; CUDA 12.8 or newer. MOSS 1024x1024 was smoke-tested on RTX 4090; capacity is uncalibrated.
- Global Volume: existing model-adapter volume mounted at `/runpod-volume`.
- Preserve min=0/max=1 and idle timeout 10 seconds.
- Leave the start command empty, or use `sh /app/workers/image/start.sh`.

The image worker uses Torch 2.9.1/cu128 and torchvision 0.24.1, constrained
separately from JoyCaption's Torch 2.8 stack. MOSS availability on an existing
endpoint requires deploying this candidate image; code registration alone
does not change the deployed endpoint. The
[2026-10-19 MOSS deployment](../../docs/deployments/2026-10-19-moss-image.md)
records the published digest and eight completed GPU requests.

CI builds this image and JoyCaption. It checks dependencies, all backend
imports, handler registration without weights, shared helpers, common-worker
tests and all four complete model suites. The model-specific folders contain
runtime implementations and reference adapters rather than release Dockerfiles.

## Adding a model

Implement a backend with a shared API profile, request normalizer and client
error types, plus `load()`, `infer(norm, seed, raw_loras)` and `close()`.
Constructors must not load weights. Register it in `image_worker.backends.REGISTRY`
and include its code/dependencies in the common image. The ownership manager,
request parsing and image delivery remain shared.

## Attribution

Model source notices and licenses remain in `../krea2/NOTICE`,
`../krea2/LICENSES/`, `../qwen_image_21/LICENSES/` and `../moss_image/LICENSES/`,
plus `../moss_image/NOTICE` and `../fibo_2/NOTICE`, and are included in the image.
