# Krea 2 Multi-Ref Identity Edit (N≤2)

**Дата:** 2026-07-28  
**Статус:** implemented (N≤2 multi-ref edit)

## Контекст

Worker `workers/krea2` уже поддерживает:

| `type` | `images` | Путь |
|--------|----------|------|
| `image_generate` | empty | text-to-image |
| `image_edit` | **exactly 1** | identity edit: grounded Qwen3-VL + source VAE tokens (RoPE frame=1) + target noise (frame=0) + optional identity LoRA / `ref_boost` |

API уже принимает `images` как **массив** (README: «future multi-ref»).  
Хелперы sampling уже multi-aware:

- `build_edit_position_ids(..., src_grids: list[tuple[int,int]], ...)` — frame = `i+1` per source  
- `build_ref_boost_bias(boosts: list[float], src_token_lens: list[int], ...)`  

Но runtime, request validation и TE жёстко single-source:

- `normalize_job_input`: `len(images) == 1`  
- `grounded_encode`: `len(texts) == len(images) == 1`, template с одним `<|image_pad|>`  
- `sample_edit` / `Krea2Pipeline.edit`: один `source: Image.Image`

Community (identity-edit LoRA v1.2, Comfy two-image workflows) использует **до двух** reference images на том же dual-conditioning path: multi-identity, multi-subject, scene+subject, edit-base+ref.

## Цели

- Разрешить `image_edit` с **1 или 2** изображениями в `images[]` (N=2 hard max в v1).
- Один compute path: расширить существующий identity-edit pipeline, **без** новых `type` и без отдельных pipeline для «same subject» vs «different objects».
- Multi-image grounded TE (оба image в vision encode + instruction).
- Multi source VAE tokens: независимо fit/crop к **одному** target canvas; cat sequence; RoPE frames 1..N; target frame 0.
- Политика размера: explicit `width`/`height`, иначе от **`images[0]`**.
- Обратная совместимость: 1 image = текущее поведение (в пределах допустимых изменений template/API surface).
- Документация: order-конвенции (scene first), примеры 3а/3б, VRAM.

## Не входит в scope

- N > 2 (жёсткий reject, не silent truncate).
- Отдельные job `type` (`image_compose`, `multi_identity`, …).
- Per-image `ref_boosts[]` (v1: один scalar `ref_boost` на **все** sources).
- Style-only dual-ref как product guarantee (режим E): тот же path технически, но не обещаем quality.
- Regional prompts / masks / ControlNet / IP-Adapter.
- `num_images != 1` для edit.
- Изменение t2i path.
- Авто-детект «same subject vs different» на worker.
- GPU end-to-end smoke в CI (unit tests only, как у edit helpers).

## Product taxonomy (один pipeline)

| Код | Сценарий | Как пользоваться |
|-----|----------|------------------|
| A | Multi-identity (несколько фоток одного субъекта) | 2 photos + prompt restage |
| B | Multi-subject (разные объекты вместе) | 2 refs + compose prompt |
| C | Scene + subject | **scene = `images[0]`**, subject = `images[1]` |
| D | Edit-base + reference | base = `images[0]`, ref = `images[1]` |
| F | Multi-view product | = A |
| E | Content + style | out of quality scope v1 |

Worker **не** ветвится по A–F: только N images + prompt + order.

## API

### Request (без нового `type`)

```json
{
  "input": {
    "type": "image_edit",
    "prompt": "create a photo of this person next to the tractor",
    "images": [
      "data:image/png;base64,...",
      "data:image/jpeg;base64,..."
    ],
    "loras": [{"name": "krea2_identity_edit_v1_2", "strength": 1.0}],
    "ref_boost": 4.0,
    "grounding_px": 768,
    "fit_mode": "fit",
    "num_inference_steps": 8,
    "guidance_scale": 0.0,
    "mu": 1.15,
    "seed": 42
  }
}
```

### Правила `images`

| Правило | Значение |
|---------|----------|
| `image_generate` | `images` empty / omitted (как сейчас) |
| `image_edit` | length **1 или 2** |
| length 0 / >2 | `RequestError` |
| decode | как сейчас: raw base64 или data-URL → RGB PIL |
| `num_images` | must be **1** (как сейчас) |

### Output size

**HTTP / `normalize_job_input` (как сегодня):**

| Raw keys в job input | Результат |
|----------------------|-----------|
| **Нет** ни `width`, ни `height` | `size_from_source=True`; `width=height=None` → canvas от **`images[0]`** |
| Есть `width` и/или `height` | schema defaults fill missing key → **оба** int после normalize; roundup ×16 |

RunPod `validate()` применяет schema defaults (`width`/`height` default 1024), поэтому «только один ключ в JSON» на wire **не** даёт half-null: после validate оба поля int, `raw_keys` помечает size as explicit.  
Invariant для client docs: prefer omit **both** for source-derived size, or send **both** explicitly. Partial send is allowed only because schema fills the other side (current behavior); we do **not** invent new partial-size semantics.

**Internal invariant (pipeline / `sample_edit`):**

| `(width, height)` | Поведение |
|-------------------|-----------|
| `(None, None)` | derive both from `sources[0]` via `target_size_from_source` |
| `(int, int)` | roundup ×16 each; use as canvas |
| **xor** (`width` set, `height` None or reverse) | **reject** with clear error — no silent “derive both from source” |

Handler always passes either both `None` or both `int` from `NormalizedRequest`.  
`sample_edit` / `Krea2Pipeline.edit` enforce the xor reject so direct library callers cannot hit the old silent path.

`images[1]` **никогда** не задаёт canvas. Разные native sizes обоих refs — **валидны**.

### Geometry per source

Для каждого source независимо:

```text
fit_source_pixels(src_i, target_h=height, target_w=width, fit_mode)
→ VAE encode → patch tokens
```

- `fit`: AR-preserving; grid может быть **меньше** target; RoPE `stride1` + center offset.  
- `crop`: exact target; RoPE `anchor`.  
- Не ресайзить img2 «как img1» до fit; не pad в pixel space до общего HxW.

### Shared edit fields (без изменений смысла)

| Field | Notes |
|-------|--------|
| `grounding_px` | long-side cap **per** image в TE |
| `ref_boost` | один scalar; attention boost target→**each** source (same value) |
| `fit_mode` | `"fit"` \| `"crop"` для **всех** sources |
| `mu` | pinned (default 1.15), auto-mu disabled |
| `loras` | как сейчас; identity LoRA рекомендуется |

### Response (backward-compatible shape)

Текущий handler кладёт edit-поля **top-level** в payload:

```json
{
  "images": ["..."],
  "image_url": "...",
  "seed": 42,
  "width": 1024,
  "height": 1024,
  "type": "image_edit",
  "loras": [...],
  "grounding_px": 768,
  "ref_boost": 4.0,
  "fit_mode": "fit"
}
```

**v1 decision (no breaking nest):**

- **Сохранить** top-level `grounding_px`, `ref_boost`, `fit_mode` (как сейчас).
- **Добавить** top-level `num_refs` (int, 1 или 2) только для `image_edit`.
- **Не** вводить nested `"edit": { ... }` в v1 (избегаем ambiguity и silent migration).
- Nested `edit` block **out of scope**; if ever added later, only as **additive** duplicate of top-level fields — never remove top-level keys.

```json
{
  "type": "image_edit",
  "grounding_px": 768,
  "ref_boost": 4.0,
  "fit_mode": "fit",
  "num_refs": 2
}
```

## Архитектура

### Data flow (N refs)

```text
images[0..N-1], prompt
    │
    ├─► grounded_encode(prompt, images[0..N-1])     # multimodal TE, multi vision tokens
    │
    ├─► for each i: fit → VAE → src_tok_i            # independent geometry
    │       RoPE frame = i+1
    │
    ├─► target noise (W×H canvas)                    # RoPE frame = 0
    │
    └─► img = cat([src_tok_0, ..., src_tok_{N-1}, tgt_tok], dim=seq)
            denoise: update only last tgt_len tokens
            sources stay clean
            ref_boost bias: target rows → each source block (same boost)
```

### Request (`request.py`)

- `NormalizedRequest.images`: `tuple[Image.Image, ...]` length 0 (generate) / **1..2** (edit).
- Edit branch: `1 <= len(images_raw) <= 2`, else `RequestError` with clear message.
- Size-from-source: primary = `images[0]` (after decode); `size_from_source` iff neither `width` nor `height` in raw keys (unchanged).
- After normalize for edit: either `(width is None and height is None)` or `(width is int and height is int)` — never xor.

### Encoder (`encoder.py`)

**Текущий MVP:**

```text
len(texts) == len(images) == 1
template: single <|vision_start|><|image_pad|><|vision_end|>
```

**v1 multi-ref:**

- Signature: `grounded_encode(texts, images_list, *, grounding_px)` где:
  - batch B=1 (CFG caller loops as today);
  - для одного sample: **один** instruction + **list** of 1..2 PIL images (не «len(texts)==len(images)» как zip single-pairs).
- Уточнение API (breaking только внутри worker, не HTTP):

```python
def grounded_encode(
    self,
    text: str,                          # single instruction
    images: Sequence[Image.Image],      # 1..2
    *,
    grounding_px: int = 768,
) -> tuple[Tensor, Tensor]:
```

  CFG negative: `grounded_encode(neg_text, images, ...)`.  
  Старый list-of-one стиль **убрать** из public helper; обновить единственного caller (`sample_edit`) и тесты template helpers.

- Template: N vision blocks перед instruction, порядок = `images` order:

```text
<|im_start|>system
Describe the image by detailing ...
<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>   # image 0
<|vision_start|><|image_pad|><|vision_end|>   # image 1 (if N=2)
{instruction}
<|im_end|>
<|im_start|>assistant
```

- `grounded_template(instruction, num_images: int = 1)` (или builder, который вставляет N vision slots).
- Каждый image: `resize_for_grounding` независимо.
- `mm_processor(text=[...], images=[img0] | [img0, img1], ...)`.
- Prefix strip: сохранить `prompt_template_encode_start_idx` behavior; multi-image **удлиняет** sequence после prefix — strip тот же text-prefix length, vision tokens остаются в returned hiddens (как community path). Проверить unit/integration consistency с single-image (regression: N=1 tensor shapes still valid).
- Reject `num_images not in (1, 2)` inside encoder.

### Sampling (`edit_sampling.py`)

```python
def sample_edit(
    ...,
    sources: Sequence[Image.Image],   # 1..2; rename from source
    width: int | None = None,
    height: int | None = None,
    ...
) -> list[Image.Image]:
```

- Validate `1 <= len(sources) <= 2`.
- Size invariant: reject if exactly one of `width`/`height` is `None` (xor); accept both None or both int.
- Target size from both int or `sources[0]`.
- TE once with all sources.
- Encode each source → list of tokens; `torch.cat` on seq dim.
- `src_grids`, `src_token_lens`, `boosts = [ref_boost] * N`.
- `build_edit_position_ids` / `build_ref_boost_bias` already multi-ready.
- Euler loop: `src_total = sum(src_lens)`; keep `img[:, :src_total]` frozen; update `[:, -tgt_len:]`.
- `mu` pin unchanged.

### Pipeline / handler

**`Krea2Pipeline.edit` public surface (worker package):**

```python
def edit(
    self,
    prompt: str,
    sources: Sequence[Image.Image] | None = None,
    *,
    source: Image.Image | None = None,  # deprecated single-ref adapter
    width: int | None = None,
    height: int | None = None,
    ...
):
```

- Preferred: `sources=` length 1..2.
- Deprecated: `source=` alone → treat as `sources=(source,)`.
- If both provided: prefer `sources` if non-empty; else error if conflicting (plan: **error if both `source` and `sources` set** — unambiguous).
- If neither set: error.
- Handler uses `sources=norm.images` only (no `source=`).
- Response meta (top-level): existing edit fields + `num_refs=len(norm.images)`.

### Tests

| Area | Cases |
|------|--------|
| `request` | edit 1 ok; edit 2 ok; edit 0 fail; edit 3 fail; generate + images fail; size_from_source uses img0 only; after normalize no xor WH |
| `encoder` helpers | template 1 pad; template 2 pads; order stable |
| `encoder` `grounded_encode` (CPU + **fake `mm_processor`**, no real weights) | 0 images reject; 3 images reject; 1 image: pad count 1, order preserved; 2 images: pad count 2, order preserved, resized count == pad count; prefix strip leaves vision-related length > 0 / tokens after strip; N=1 regression shape path |
| `edit_sampling` pos/bias | two src grids different sizes; frames 1,2,0; boost spans both src blocks |
| `edit_sampling` size | xor width/height raises; both None derives from sources[0] (mock/pure path if extractable); both int accepted |
| `pipeline.edit` | `source=` alone still works (deprecated adapter); `sources=` preferred; both set → error |
| regression | existing single-ref pos/bias tests still pass |

No full GPU model load / real Qwen weights in unit tests. Fake processor must assert call kwargs (`images` list length/order, text contains N vision slots).

### Docs (`README.md`)

- `images` length 1..2 for edit.
- Examples: single edit; scene+person (order); multi-identity restage.
- Size policy; VRAM note (~3× image tokens with 2 refs).
- Out of scope: N>2, style dual-ref guarantee.

## Error messages (safe client-facing)

- `image_edit requires 1 or 2 entries in images[]`  
- width/height xor (library path): e.g. `width and height must both be set or both omitted`  
- pipeline: `edit requires sources= or source=` / `pass only one of source= or sources=`  
- Existing: multiples of 16, fit_mode, grounding_px, ref_boost, num_images==1  

## VRAM / ops

- Sequence length ≈ text + sum(source tokens) + target.  
- 2 refs ≈ ~3× image tokens vs t2i; vs 1-ref edit ≈ +1× source.  
- Same 24 GB class guidance: ≤1.5 MP target, TE offload, prefer `fit`, avoid CFG when tight.  
- OOM → same recovery advice as current edit README.

## Acceptance criteria

1. `image_edit` + 1 image: HTTP response still has top-level `grounding_px` / `ref_boost` / `fit_mode`; plus new top-level `num_refs: 1`.  
2. `image_edit` + 2 images: validation accepts; `num_refs: 2`; both sources in TE + VAE path (unit-covered structure / fake processor).  
3. Different native sizes for img0/img1 accepted; canvas from explicit both-int size or img0.  
4. 0 or ≥3 images → clear `RequestError`.  
5. xor `width`/`height` on `sample_edit` / pipeline → clear error (no silent dual-derive).  
6. `Krea2Pipeline.edit(source=...)` still works as deprecated adapter.  
7. Unit tests green without GPU (including fake-processor `grounded_encode`).  
8. README updated; no nested `edit` response object.

## Open decisions (resolved in discussion + review)

| Topic | Decision |
|-------|----------|
| Separate pipelines 3а/3б | **No** |
| Max N | **2** |
| Canvas primary | **`images[0]`** or explicit both WH |
| Partial WH | schema fills missing key; library rejects xor |
| Per-ref boost | **v1 no** (shared `ref_boost`) |
| New job type | **No** |
| Style dual-ref product | **Out of scope quality-wise** |
| Response shape | **top-level only**; add `num_refs`; no nested `edit` in v1 |
| `source=` kwarg | **keep** as deprecated adapter; prefer `sources=` |

## Implementation order (for plan)

1. Request validation + tests (incl. size_from_source / no xor after normalize)  
2. Encoder multi-image template + `grounded_encode` API + helper tests + **fake mm_processor tests**  
3. `sample_edit` multi-source + position/bias + xor size tests  
4. Pipeline `sources=` + deprecated `source=` + handler top-level `num_refs`  
5. README  
6. Commits per slice (TDD)

## Risks

| Risk | Mitigation |
|------|------------|
| Prefix strip wrong with 2 vision blocks | Same text-prefix index; N=1 regression; fake-processor assert post-strip length |
| Processor multi-image API quirks | Match community template; fail loud; **CPU fake-processor unit tests** for call contract |
| Silent size xor in library callers | Explicit reject in `sample_edit` / pipeline |
| Response breaking nest | Stay top-level; additive `num_refs` only |
| Quality order sensitivity | Document scene-first; no auto reorder |
| VRAM OOM on 2 large crop refs | Docs + existing MP cap on target; fit default |
