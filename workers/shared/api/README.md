# Image worker API: Krea 2, Qwen Image 2.1 and MOSS-Image

All image backends use this shared request parser, common request fields, seed selection,
image decoding/delivery, response builder and error handling. Their model adapters
own inference, model selection, reference geometry and LoRA compatibility.
`profiles.py` declares defaults, limits and supported model options.

The common RunPod endpoint is `9zb0wyo61ck3wk` (`image-workers`).
Set `input.model` to `krea2`, `qwen_image_21` or `moss_image`; omission defaults to Krea.
Qwen and MOSS Base/Turbo are selected with `model_options.variant` on the same endpoint.
MOSS requires a worker image including its backend; see its section below.
The response reports the selected model in `output.model`. Unknown model IDs
are rejected before loading or unloading a model.

## Request

Submit `POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run` with an
`Authorization: Bearer <RUNPOD_API_KEY>` header and the following JSON envelope.
The returned request ID is polled at `/status/{REQUEST_ID}` by the bundled clients.

```json
{
  "input": {
    "model": "qwen_image_21",
    "type": "image_generate",
    "prompt": "A photograph of a red fox in soft evening light",
    "negative_prompt": "blur, artifacts",
    "width": 1024,
    "height": 1024,
    "seed": 123,
    "num_inference_steps": 40,
    "guidance_scale": 3.5,
    "sampler": "euler",
    "num_images": 1,
    "images": [],
    "loras": [],
    "model_options": {"variant": "base"}
  }
}
```

This example selects the Qwen backend on the common endpoint. `prompt` is required; all other fields
are optional. Unknown root fields and unknown model options are errors.

| Common field | Type | Krea 2 | Qwen Image 2.1 |
|---|---|---|---|
| `type` | string | `image_generate` (default), `image_edit` | Same |
| `prompt` | string | Nonblank, required | Same |
| `negative_prompt` | string or null | Optional; used when guidance > 0 | Optional; empty unconditional prompt supplied when CFG > 1 and text omitted/blank |
| `width`, `height` | integer | Explicit values: 256..2048, multiples of 16 | Explicit values: 256..2752, multiples of 32; area <= 4,300,800 pixels |
| `seed` | integer or null | Omitted/null: random 32-bit seed; explicit seed, including 0, preserved | Same |
| `num_inference_steps` | integer | 1..64, default 8 | 1..200, default Turbo 8 / Base 40 |
| `guidance_scale` | number | Finite Krea guidance offset, default 0; <= 0 disables CFG | Finite value >= 0, default 1; <= 1 disables CFG |
| `sampler` | string | `euler` only, default `euler` | `euler` (default), `heun`, `er_sde` |
| `num_images` | integer | Generate: 1..4 (default 1); edit: exactly 1 | Same |
| `images` | array of strings | Generate: absent/empty; edit: 1..2 references | Generate: absent/empty; edit: 1..10 references |
| `reference_megapixels` | number or `"max"` | Default `"max"`; total reference budget and automatic edit canvas area | Same |
| `loras` | array of objects or null | Up to 4 adapters; default [] | Same count; supported adapter types differ below |
| `model_options` | object | Krea-specific options below; default {} | Qwen-specific options below; default {} |

Seed integers follow Torch's accepted range: `-9223372036854775808` through
`18446744073709551615`. Use a nonnegative 32-bit seed for portable clients.
Numbers cannot be booleans; floating-point NaN/infinity are rejected.

Each reference is raw base64 or a `data:image/...;base64,...` URL, never a remote
image URL. Input order is preserved. Krea converts references to RGB. Qwen keeps
alpha when present, limits each decoded file to 25 MiB and each source side to
8192 pixels, then downsizes references using the megapixel limit below.

### MOSS generation contract

MOSS uses the same envelope and response/delivery lifecycle. The Krea/Qwen
table above describes their existing contracts; MOSS differences are:

- Only `image_generate`; `images` must be empty and user `loras` must be empty/null.
- `model_options.variant`: `turbo` (default) or `base`. Both use MOSS-Image-SFT;
  Base here means SFT without the accelerator, not MOSS-Image-Base weights.
- Default steps/CFG: Turbo 8/1; Base 50/3. Steps may be overridden in 1..200.
- Only Euler. Base's negative branch runs when CFG differs from exactly 1;
  Turbo requires CFG exactly 1 and no nonblank negative prompt.
- Dimensions: 256..4096, multiples of 16, area <= 4,194,304 pixels; default 1024x1024.
- `model_options.shift`: positive finite number, default 5.
- `model_options.uncond_skip_layers`: unique indices 0..41; omitted/null selects
  `[22]` for Base or `[]` for Turbo. An explicit empty list disables skipping.
  Turbo rejects nonempty skip layers.

Turbo applies the pinned built-in DMD adapter unmerged at strength 0.7; it is
reported in response `loras` as diagnostic metadata, not as a user adapter to
copy into the next input. Switching modes retains the SFT pipeline and removes
the adapter after every Turbo call, including failures. The response includes
`model_revision` and, for Turbo, `lora_revision`.

`reference_megapixels` is accepted by the common parser but does not choose
MOSS's generation canvas or establish a calibrated capacity. Plain prose,
tags and serialized JSON layouts go directly into `prompt`.
See [MOSS runtime and examples](../../moss_image/README.md).

### Krea/Qwen guidance semantics and sampling

Existing guidance behavior is preserved:

- Qwen: `uncond + guidance_scale * (cond - uncond)` when guidance > 1.
- Krea: `cond + guidance_scale * (cond - uncond)` when guidance > 0.

Thus Krea guidance 2.5 corresponds to standard CFG 3.5. Moving to the shared
contract does not silently change existing Krea guidance values. In both models,
an empty negative prompt is usable; CFG does not require explicit negative text.

Krea uses its existing Euler flow sampler. Requesting `heun`/`er_sde` on Krea is
a validation error. API extraction does not add those solvers to Krea.

Qwen Base uses the ComfyUI QwenImage21 `simple` sigma grid. Turbo at exactly
8 steps uses the official checkpoint's saved sigma grid and transforms. Other
Turbo step counts explicitly use a simple grid of the requested length; that
override is experimental and does not guarantee improved quality.
Euler and ER-SDE use one denoiser evaluation per step; Heun uses `2 * steps - 1`.
Enabled CFG doubles the number of model passes. Qwen ER-SDE additional noise uses
the request's seeded generator. Changing sampling options does not reload weights.

### Shared megapixel budget

`reference_megapixels` belongs at the **top level** of `input`, alongside
`images` and `num_inference_steps`. Both workers accept a number or `"max"`
(the default). Numeric values are finite, 0.001024..4.3008 MP, where 1 MP means
1,000,000 pixels. Existing model canvas limits still apply.

With `"max"`, the worker reads **total VRAM of the active CUDA device** and
selects a budget from [`MEGAPIXEL_LIMITS`](../megapixels.py). There are separate
rows for Qwen Base/Turbo and Krea FP8/INT8 ConvRot. Total CUDA memory is converted
from binary GiB to decimal GB and rounded up to the next whole GB before selecting
the largest nominal GB tier not exceeding that capacity. For example, L40S
reports 44.40 GiB / 47.68 GB and selects the 48 GB tier. Missing CUDA, unknown
models and capacities below the table use the 1 MP fallback.

**The 48 GB tier selects 4 MP** for all four profiles, calibrated on L40S with
Krea Identity Edit V2 / Qwen Lenovo as fixed LoRA baselines. All other tiers
remain 1 MP placeholders, including larger GPUs pending calibration.
See the [measurements, scope and repeatable procedure](../../../docs/benchmarks/image-megapixels.md).
Additional LoRAs, reference counts, aspect ratios and CFG can change memory use;
the current lookup only considers model and total VRAM. Explicit numbers bypass the table
and GPU detection. Free/remaining VRAM is never used, so prior jobs do not alter
the selected tier.

The selected budget is divided equally among input references; unused shares
are not redistributed. References are only downscaled, with model-specific
alignment. The automatic edit canvas gets the **full** selected area, independent
of reference count, including upscaling small inputs. Its aspect comes from the
last original reference on Qwen and the first original reference on Krea.
Explicit width/height (or Qwen `output_resolution`) override the automatic canvas,
while reference resizing still uses the budget. Text-to-image size behavior is unchanged.

Responses return `reference_megapixels` with the requested value (including
`"max"`), `resolved_reference_megapixels` with the chosen number, and `vram_gib`
with the detected capacity (null for explicit numbers or unavailable CUDA).
These values are also logged. When repeating a request, reuse the common
`reference_megapixels` field separately from `model_options`.

The old `model_options.reference_megapixels` input is accepted for compatibility
and normalized to the top level. Conflicting values in both locations are errors.

```json
{"input": {"type": "image_edit", "prompt": "Combine the references", "images": ["data:image/png;base64,..."], "reference_megapixels": "max"}}
```

### Qwen `model_options`

| Key | Type | Default | Behavior |
|---|---|---|---|
| `variant` | string | `turbo` | `turbo` or `base`; one resident pipeline, unloaded before switching variants |
| `output_resolution` | integer or null | null | 256..2752, multiple of 32. Sets edit canvas area when width/height are omitted; preserves the last reference's aspect ratio |

Generate defaults to 1024x1024. For edit, omit both width and height to target
`reference_megapixels` at the **original last reference's aspect ratio**, even
when the source is small. Dimensions round down to multiples of 32, so the
area is approximately the requested MP and never exceeds it. For example,
100x30 produces a 1824x544 canvas (0.992256 MP) by default. Pass both dimensions
for an explicit canvas; passing only one is an error. Each reference receives a
pixel cap of `reference_megapixels * 1,000,000 / number_of_references` and is downscaled
on CPU before the vision/text encoder and VAE, preserving aspect ratio subject to
32px alignment. Dimensions round down to multiples of 32 (minimum 32px), so the
inference area stays below the budget. Smaller references are not enlarged, apart
from the model's 32px minimum side. For example, a 2048x2048 or 1024x1024 reference
becomes 992x992 when it is the only reference at the default 1 MP budget.
Two square references become 704x704 each, five become 416x416 each, while the
automatic square output remains 992x992 in all three cases. Unused area from small
references is not redistributed. The budget must allow the 32x32 minimum for every
reference; smaller budgets are validation errors. References also fit 2752px per side.
This limits the combined reference area; GPU memory also depends on the output
canvas, CFG, LoRAs and model variant. It does not guarantee that every request fits a GPU.

Explicit dimensions or `output_resolution` determine the output canvas separately
from the reference budget. The canvas must fit 256..2752px per side and 4,300,800
pixels; unsupported aspect-ratio/area combinations are validation errors.
In response `model_options`, `output_resolution: null` preserves the automatic
mode so the options can be reused for another edit. The legacy top-level
`output_resolution` also reports the derived control value. VAE choice, offload
and mmap remain worker configuration, not request fields.

```json
{
  "input": {
    "type": "image_edit",
    "prompt": "Keep the subject and change the background to a sunset beach",
    "images": ["data:image/png;base64,..."],
    "num_inference_steps": 40,
    "guidance_scale": 3.5,
    "sampler": "er_sde",
    "seed": 123,
    "loras": [{"name": "portrait_style", "strength": 0.8}],
    "reference_megapixels": "max",
    "model_options": {"variant": "base"}
  }
}
```

### Krea `model_options`

| Key | Type | Default | Behavior |
|---|---|---|---|
| `mu` | number or null | 1.15 | Finite time-shift parameter. null requests resolution-dependent shift for generate; edit resolves null to 1.15 |
| `grounding_px` | integer | 768 | Vision text-encoder long-side cap per reference; 0 preserves native size; otherwise 256..2048 |
| `ref_boost` | number | 1.0 | Finite value >= 0; attention boost toward last/only reference |
| `ref_boost_a` | number | 1.0 | First reference boost in two-image workflows |
| `ref_position` | string | `aligned` | Last/only reference placement: `aligned`, `beside`, `bag_of_features` (experimental) |
| `ref_position_a` | string | `aligned` | First reference placement in two-image workflows; same choices |
| `fit_mode` | string | `fit` | `fit` preserves aspect while fitting each source; `crop` uses center crop to target aspect |

Generate defaults to 1024x1024. Edit with neither dimension uses the **first**
reference's aspect ratio at the shared resolved megapixel budget and aligned to
16. Explicit dimensions define the
target canvas. For legacy compatibility, passing only one dimension still uses
the schema default 1024 for the other; new clients should pass both or neither.
The other edit options have no effect on text-to-image generation.

```json
{
  "input": {
    "type": "image_edit",
    "prompt": "Place both people together in a garden",
    "images": ["data:image/png;base64,...", "data:image/png;base64,..."],
    "num_inference_steps": 8,
    "guidance_scale": 0.0,
    "sampler": "euler",
    "seed": 123,
    "loras": [
      {"name": "krea2_identity_edit_v2", "type": "lora", "strength": 1.0},
      {"name": "identity_delta", "type": "weight_diff", "strength": 4.0}
    ],
    "model_options": {"mu": 1.15, "grounding_px": 768, "ref_boost": 2.0, "ref_boost_a": 1.0,
                      "ref_position": "beside", "ref_position_a": "aligned", "fit_mode": "fit"}
  }
}
```

Krea downloads Identity Edit V2 and General Edit V2 on its first request. Select
`krea2_identity_edit_v2` for people/restaging or `krea2_general_edit_v2` for
instruction edits. Use one at strength 1.0; combining both at 1.0 produces noise.
The worker does not silently insert an adapter when `loras` is empty.
The CLI's `--edit-model identity|general` selects the adapter and recommended
boost/placement defaults. These controls require the updated Krea image.

### LoRA entries

```json
{"name": "adapter_id", "type": "lora", "strength": 1.0}
```

| Key | Rules |
|---|---|
| `name` | Required catalog ID; filename stem without `.safetensors`; paths and duplicate names rejected |
| `type` | Default `lora`. Krea accepts `lora` and `weight_diff`; Qwen accepts only `lora` |
| `strength` | Default 1.0; finite number. `lora`: 0..2; Krea `weight_diff`: 0..5 |

Unknown LoRA keys are errors. Adapters must exist in the worker's configured
LoRA directory (or Krea's built-in V2 catalog) and be compatible with that model. A zero-strength adapter is
validated but not applied. Selected files are copied/repaired locally as needed,
including on a warm worker; unrelated adapters are not copied for the request.
Adapters are removed after inference, including after failures.

## Response

RunPod wraps the worker result in `output`. Both workers return:

```json
{
  "images": ["data:image/png;base64,..."],
  "seed": 123,
  "width": 1024,
  "height": 1024,
  "type": "image_edit",
  "num_inference_steps": 40,
  "guidance_scale": 3.5,
  "sampler": "euler",
  "loras": [],
  "reference_megapixels": "max",
  "resolved_reference_megapixels": 1.0,
  "vram_gib": 23.7,
  "model_options": {"variant": "base", "output_resolution": null},
  "num_refs": 1
}
```

- `images`: generated PNG data URLs, or storage URLs when `BUCKET_ENDPOINT_URL`
  is configured. Upload failures fall back to PNG data URLs.
- `seed`: actual seed used; `width`/`height`: actual first output image dimensions.
- `num_inference_steps`, `guidance_scale`, `sampler`: resolved request settings.
- `loras`: applied selection with explicit `name`, `strength`, `type`.
- `model_options`: normalized model-specific controls, including defaults. Qwen
  keeps null for automatic `output_resolution`; Krea includes all its options
  and resolves edit null mu.
- `num_refs`: edit only, number of reference images.

For compatibility, Qwen also returns `model_variant`, `model_id`, `scheduler`
(`simple` or `checkpoint`), and `output_resolution` at the top level.
Krea edit also retains top-level `grounding_px`, `ref_boost`, and `fit_mode`.

## Errors and compatibility

Invalid inputs return `{"error": "..."}` without invoking inference or requesting
a restart. This covers malformed job envelopes, unknown fields/options,
unsupported samplers, type/range errors, invalid references and LoRA errors.
Missing files return an error without automatic restart. CUDA OOM and unexpected
inference exceptions retain `refresh_worker: true`; unexpected exceptions also
include a traceback. These are worker `output` values inside RunPod's envelope.

Old flat model-specific keys remain supported:

| Old top-level key | Canonical nested key |
|---|---|
| Qwen `model_variant` | `model_options.variant` |
| Qwen `output_resolution` | `model_options.output_resolution` |
| Krea `mu`, `grounding_px`, `ref_boost`, `ref_boost_a`, `ref_position`, `ref_position_a`, `fit_mode` | Same names inside `model_options` |

Aliases are compared **before** defaults are inserted. Both forms may appear
together when their values agree. Different values are a validation error;
neither form silently overrides the other. Omitted fields retain model defaults
and existing CFG/geometry behavior. Declared field limits are now checked in the
shared parser instead of relying on the RunPod SDK validator's default handling.

The CLI flags remain compatible. Both clients now send model-specific fields
under `model_options`; both expose `--steps`, `--guidance-scale`/`--cfg`, and
`--sampler` with their worker's supported choices.
Both CLIs accept `--reference-megapixels max` (default) or a number such as `0.5`
and send it at the top level. The shared parameter and `max` require updated images.
Canonical nested requests require the updated worker images; old images accept
the legacy flat form. Existing clients using the flat form work with new images.

## Implementation and verification

- `contract.py`: schemas, aliases, field validation and common normalized request.
- `profiles.py`: Krea/Qwen capabilities and model-specific option schemas.
- `images.py`: decoding, alpha policy and PNG delivery.
- `handler.py`: seed selection, inference/delivery lifecycle, response and errors.
- Worker handlers: model adapters; model request modules: geometry policy.

Shared API tests need Pillow and the standard library, without Torch, GPU or
downloaded checkpoints:

```bash
python -m unittest discover -s workers/shared/tests -p 'test_*.py'
python -m unittest discover -s workers/krea2/tests -p 'test_api*.py'
```

Worker suites additionally exercise model-specific contracts and, when installed,
the actual Torch/Diffusers CPU runtime. CPU checks do not prove GPU output quality.
