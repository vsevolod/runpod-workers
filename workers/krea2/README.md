# RunPod worker — Krea 2 Turbo FP8

Thin serverless worker: `prompt` → PNG (base64). No ComfyUI.

| | |
|--|--|
| **Model** | [AlperKTS/Krea2_FP8](https://huggingface.co/AlperKTS/Krea2_FP8) DiT + official [krea-ai/krea-2](https://github.com/krea-ai/krea-2) sampler |
| **Text encoder** | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (HF, bf16) |
| **VAE** | Qwen-Image VAE (local `qwen_image_vae.safetensors` or HF) |
| **GPU** | **24 GB** recommended (all modules resident) |
| **Template patterns** | [worker-sdxl](https://github.com/runpod-workers/worker-sdxl) |

## Layout

```
workers/krea2/
├── Dockerfile              # build context = monorepo root
├── handler.py              # RunPod entry + ModelHandler load-once
├── schemas.py
├── download_weights.py     # volume bootstrap (not baked into image)
├── test_input.json
├── requirements.txt
└── krea2_infer/            # vendored krea-2 + FP8 loader
```

## Network volume

Mount the same datacenter volume on the endpoint. Example:

```text
/runpod-volume/krea2/
  krea2_turbo_fp8.safetensors     # required (~12 GB)
  qwen_image_vae.safetensors      # optional if HF VAE cache is warm
/runpod-volume/hf/                # optional HF_HOME for offline TE/VAE
```

Bootstrap on a Pod with the volume attached:

```bash
pip install huggingface-hub hf_transfer
export HF_TOKEN=hf_...   # if needed
export HF_HOME=/runpod-volume/hf
python download_weights.py --output /runpod-volume/krea2 --hf-home /runpod-volume/hf
```

> **Note:** Comfy-oriented `qwen3vl_4b_fp8_scaled.safetensors` is **not** used by this worker. Text encoding goes through Hugging Face `Qwen3VLForConditionalGeneration` (official krea-2 path). Cache that model on the volume via `download_weights.py` so cold starts do not re-download ~8 GB.

## Endpoint env

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_DIR` | `/runpod-volume/krea2` | DiT / optional VAE files |
| `DIT_PATH` | auto under `MODEL_DIR` | Override DiT safetensors path |
| `VAE_PATH` | auto under `MODEL_DIR` | Override VAE safetensors path |
| `TEXT_ENCODER_ID` | `Qwen/Qwen3-VL-4B-Instruct` | HF id or local snapshot path |
| `HF_HOME` | (hf default) | Put on volume for faster/offline loads |
| `LOCAL_FILES_ONLY` | unset | `1` after cache is warm |
| `BUCKET_ENDPOINT_URL` | unset | If set, upload via RunPod S3 helpers instead of base64 |

## Deploy (RunPod)

1. Create Network Volume (≥40 GB free: FP8 DiT + TE + VAE + slack).
2. Run `download_weights.py` on a Pod with that volume.
3. **Serverless → New Endpoint → GitHub**
   - Dockerfile: `workers/krea2/Dockerfile`
   - GPU: **24 GB** class
   - Attach the volume
   - Container disk: 20+ GB
   - FlashBoot: on
   - Active workers: `0` (or `1` for warm)
   - Env: `MODEL_DIR=/runpod-volume/krea2`, ideally `HF_HOME=/runpod-volume/hf`

## API

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

Turbo defaults: **8 steps**, **CFG 0**, **mu 1.15**. Width/height multiples of **16** (1024–2048).

Response:

```json
{
  "output": {
    "images": ["data:image/png;base64,..."],
    "seed": 42,
    "width": 1024,
    "height": 1024
  }
}
```

### curl

```bash
export RUNPOD_API_KEY=rpa_...
export ENDPOINT_ID=...

curl -sS -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d @workers/krea2/test_input.json
```

## Local image build

From monorepo root:

```bash
docker build -f workers/krea2/Dockerfile -t runpod-krea2:latest .
```

You still need GPU + `MODEL_DIR` mount to actually generate.

## Design notes

- **All resident (24 GB):** TE + DiT + VAE stay on GPU. No TE unload in this MVP.
- **FP8 DiT:** weights stay `float8_e4m3fn` where quantized; Linear layers cast to bf16 on the fly so VRAM stays closer to ~12 GB for the transformer.
- **Not in MVP:** LoRA, Comfy workflows, baking 18 GB into the image layer.

## License

- Worker scaffolding: use under the same terms as your deployment; SDXL worker inspiration is MIT upstream.
- Krea 2 weights / community license: https://www.krea.ai/krea-2-licensing  
  Commercial use may require contacting opensource@krea.ai.
