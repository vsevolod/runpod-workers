# Fibo-2 generation and editing

Fibo uses the common image worker and endpoint `9zb0wyo61ck3wk`.
Select `input.model: "fibo_2"`. Availability requires an image containing this
backend and an `HF_TOKEN` authorized to download both gated Bria checkpoints.
No model weights or VGL conversion service are bundled.

| API variant | Upstream checkpoint | Capabilities | Default steps / CFG |
|---|---|---|---|
| `base` (default) | [briaai/Fibo-2.0-dev](https://huggingface.co/briaai/Fibo-2.0-dev) | Generation and editing | 30 / 5 |
| `turbo` | [briaai/Fibo-2.0-edit-turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo) | Editing only | 4 / 1 |

These are separate full checkpoints. Switching variants releases the previous
pipeline before loading the selected one. Source revisions are pinned in
[`config.py`](fibo_2_infer/config.py); each variant has its own cache and manifest.
The native Diffusers implementation is pinned in the common image requirements.

## VGL JSON API

`prompt` must be a nonempty **JSON object**, rather than a string containing JSON.
The worker serializes it only at the native model boundary. It does not call an
LLM, OpenRouter, Gemini or `bria-vgl`. Use the model's VGL structure to describe
the scene; the smallest supported generation caption has `short_description`.
Additional VGL fields pass through unchanged. Structural checks do not constitute
full semantic validation of every VGL field.

Generation:

```json
{
  "input": {
    "model": "fibo_2",
    "type": "image_generate",
    "prompt": {"short_description": "A red fox sitting in a snowy forest."},
    "model_options": {"variant": "base"},
    "width": 1024,
    "height": 1024,
    "seed": 0
  }
}
```

Editing requires 1..4 ordered base64/data-URL references and a nonblank
`edit_instruction`. Supply only the instruction, or also a target caption:

```json
{
  "input": {
    "model": "fibo_2",
    "type": "image_edit",
    "prompt": {
      "structured_caption": {"short_description": "The same fox with blue fur in a snowy forest."},
      "edit_instruction": "Change the fox's fur to blue; preserve the composition."
    },
    "images": ["data:image/png;base64,..."],
    "model_options": {"variant": "turbo"},
    "seed": 0
  }
}
```

Base accepts an optional `negative_prompt` JSON object. Omission/null preserves
the native trained null caption; it is not replaced with an empty string.
Turbo requires CFG exactly 1 and rejects any negative prompt. Both variants
reject additional LoRAs and support only the native flow-match Euler scheduler.
Steps can be overridden in 1..200. Guidance is a finite number >= 0.
Generation supports `num_images` 1..4; editing returns exactly one image.

Explicit width and height must be supplied together: multiples of 16,
256..4096 per side, total area <= 4,194,304 pixels. Generation defaults to
1024x1024. Editing without dimensions preserves the first source's aspect at
the shared `reference_megapixels` area, rounding down to 16px alignment.

The `"max"` reference budget is currently **1.048576 MP on every GPU**, matching
the native 1024-square default. It is an uncalibrated default, not a measured
VRAM limit. For multiple references the budget is shared equally; references
are only downscaled, with 16px alignment and a 4096px side cap. Native Fibo
aligns a single reference to the output canvas, so its canvas must also fit
the reference budget; increase `reference_megapixels` for a larger explicit
single-reference edit. These limits do not guarantee that a request fits a GPU.

The shared response includes `model`, dimensions, seed, steps, guidance,
reference budget and image delivery fields. Fibo additionally reports
`model_id`, immutable `model_revision`, `model_variant` and scheduler.

## Client

Store the prompt object alone (without the `input` envelope) in a JSON file:

```bash
python scripts/fibo_2.py --prompt-file scene.json --seed 0 -o fox.png
python scripts/fibo_2.py --prompt-file edit.json --image fox.png --variant turbo -o blue.png
python scripts/fibo_2.py --request-id JOB_ID -o result.png
```

Repeat `--image` to send ordered references. `--negative-prompt-file` accepts a
Base negative VGL object. `--steps`, `--cfg`, `--width`, `--height`,
`--reference-megapixels` and `--num-images` override the shared API defaults.

## Loading and verification

The shared CPU-only downloader stages transformer -> text encoder/processor/fonts
-> VAE/scheduler, publishing each component after expected-size validation.
Native weights load locally in BF16 while later components download. Final
`all` readiness and successful downloader exit are required before inference.
Interrupted retries retain completed pinned components and repair partial files.
Fonts and their license files are staged with the processor, before text rendering.

`FIBO_MODEL_DIR` defaults to `/models/fibo_2`, with `base/` and `turbo/` children.
`FIBO_OFFLOAD=cpu` enables native component CPU offload; `none` places the pipeline
on CUDA. Each snapshot needs approximately 25.5 GB, or 51 GB for both variants,
besides other model caches and container layers. GPU/host-RAM requirements and
cold-start improvement have not been measured by the CPU tests.

Run its suite with:

```bash
python -m unittest discover -s workers/fibo_2/tests -p 'test_*.py'
```

Tests cover JSON input, routing, variant ownership, partial-file repair, pinned
cache reuse and actual tiny local BF16 model loads with a controlled downloader
subprocess. These tests do not establish production GPU image quality or capacity.
