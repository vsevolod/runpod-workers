# Krea 2 LoRA `type` + weight_diff

**Дата:** 2026-07-26  
**Статус:** согласовано для планирования реализации

## Контекст

Worker `workers/krea2` уже поддерживает до четырёх runtime text-to-image LoRA
из allowlist на Network Volume (`LORA_DIR/*.safetensors`). API job-а принимает
только `{name, strength}`; loader ждёт low-rank пары A/B (или up/down) и
применяет `up(down(x)) * (alpha/rank) * strength`. Strength capped `0.0..2.0`.

Community-адаптеры вроде [Krea2 Filter Bypass Fedor](https://civitai.com/models/2746817)
не являются rank-LoRA. Это full-matrix delta к
`txtfusion.projector.weight` (`nn.Linear(num_txt_layers, 1)`, shape `[1, 12]`),
хранимая как safetensors-ключ `diffusion_model.txtfusion.projector.diff` с
ненулевыми «knob»-колонками 9 и 10. Рекомендуемый strength — **3–5**.

Текущий loader отклонит такой файл (`unsupported tensor key`). Нужен явный
дискриминатор формата и отдельный runtime-path, без хранения адаптеров в git
и без передачи payload в request.

## Цели

- Добавить optional поле `type` в каждый элемент `loras[]`.
- Default `type` — `"lora"` (обратная совместимость с текущими клиентами).
- Поддержать второй type `"weight_diff"`: full weight delta на `nn.Linear`.
- Strength range **зависит от type**: `lora` → `0.0..2.0`, `weight_diff` → `0.0..5.0`.
- Файлы по-прежнему только на volume; клиент выбирает ID + type + strength.
- Type **не** угадывается молча: loader валидирует, что содержимое файла
  соответствует заявленному type.
- В успешном ответе эхо нормализованных `{name, strength, type}`.
- Сохранить lifecycle: load CPU → activate на denoise → deactivate до VAE;
  process-local lock; max 4 adapters; mix `lora` + `weight_diff` allowed.

## Не входит в scope

- Download-by-URL, base64 payload, inline weights.
- TE/VAE adapters.
- Auto-detect type без явного/default.
- LyCORIS / LoHa / LoKr / DoRA.
- Отдельный top-level field вроде `bypass` / `projector_delta`.
- Хранение recommended strength, trigger words или иных клиентских метаданных
  в worker.
- Помещение `fedor_bypass.safetensors` (или любых user LoRA) в git/репозиторий.
- Автоматический GPU smoke test в CI.

## API

### Request

```json
{
  "input": {
    "prompt": "portrait, cinematic light",
    "loras": [
      {"name": "style_v1", "strength": 0.85},
      {"name": "fedor_bypass", "type": "weight_diff", "strength": 4.0}
    ]
  }
}
```

Поля элемента `loras[]`:

| Поле | Обязательно | Default | Правила |
|------|-------------|---------|---------|
| `name` | да | — | allowlist ID (stem файла), как сейчас |
| `strength` | нет | `1.0` | finite number; range по `type` |
| `type` | нет | `"lora"` | только `"lora"` \| `"weight_diff"` |

Общие правила (без изменений, кроме strength ceiling):

- не более 4 элементов;
- уникальные `name`;
- имена только через catalog resolve (no path/URL/suffix);
- boolean не является числом;
- `strength: 0.0` → adapter skip (файл не читается, не попадает в applied list);
- лишние ключи в объекте → `LoRAError`.

Strength ceilings:

| `type` | min | max |
|--------|-----|-----|
| `lora` | 0.0 | 2.0 |
| `weight_diff` | 0.0 | 5.0 |

Неизвестный `type` или strength вне range своего type → client-facing
`LoRAError` до генерации.

### Response

Успешный `output.loras` всегда включает нормализованный `type`:

```json
{
  "loras": [
    {"name": "style_v1", "strength": 0.85, "type": "lora"},
    {"name": "fedor_bypass", "strength": 4.0, "type": "weight_diff"}
  ]
}
```

Даже если клиент не передал `type`, в ответе будет `"type": "lora"`.

## Доставка файлов

Без изменений модели volume:

```text
$LORA_DIR/
  style_v1.safetensors
  fedor_bypass.safetensors
```

- Startup catalog scan: `*.safetensors` в корне → ID = stem.
- Catalog **не** знает type и не читает тензоры.
- После добавления/удаления файлов тёплый worker нужно перезапустить.
- Репозиторий worker’а не содержит user LoRA files.

## Компоненты

### `normalize_lora_requests`

Расширить допустимые ключи объекта: `name`, `strength`, `type`.

- `type` optional string; default `"lora"`.
- Allowed set: `{"lora", "weight_diff"}`.
- После нормализации strength проверять ceiling **по type**.
- `LoRASelection` получает поле `type: str`.
- `as_dict()` возвращает `{name, strength, type}`.

### `LoRALoader`

`load` / `_load_one` ветвятся по `selection.type`.

#### type `"lora"` (существующее поведение)

- Разрешены только pair-суффиксы `lora_A`/`lora_B`, `lora_down`/`lora_up` и
  scalar `alpha`.
- Ключи `.diff` (или иные non-pair) → `Invalid LoRA …: unsupported tensor key`
  / mismatch for type.
- Shape/rank/alpha validation без изменений.
- Результат: `PreparedLoRA` со слоями low-rank.

#### type `"weight_diff"`

- Разрешены ключи, оканчивающиеся на `.diff` (после strip component prefixes
  `diffusion_model.` / `transformer.`).
- Base name до `.diff` резолвится тем же target mapping, что и LoRA
  (в т.ч. `txtfusion.projector` / `text_fusion.projector` → native
  `txtfusion.projector`).
- Tensor: floating, finite values, `ndim == 2`, shape ==
  `module.weight.shape` для target Linear.
- Не optional alpha; нет A/B.
- Хотя бы один валидный diff; unknown target / shape mismatch / non-diff keys
  → reject entire adapter (no partial apply).
- Результат: `PreparedLoRA` (или узкий sibling dataclass) со слоями
  weight-diff: target + delta tensor + strength later applied as multiplier.

Файлы **не** auto-detect: если `type="lora"`, а в файле только `.diff` —
ошибка; если `type="weight_diff"`, а в файле A/B — ошибка.

### Runtime Linear

Сохранить monopatched forward:

```text
y = base_linear(x)   # FP8 cast path unchanged
y += Σ  up(down(x)) * (alpha/rank) * strength     # type lora
y += Σ  linear(x, delta) * strength               # type weight_diff
```

- Base `Parameter` не мутируется и не fuse’ится.
- Diff tensors: BF16 на device DiT только внутри activation window.
- Activation/deactivation lifecycle, process-local lock, cleanup before VAE —
  общие для обоих types.
- Можно миксовать до 4 adapters любых types в одном job.

Минимальная модель данных runtime-слоя:

- low-rank: `down`, `up`, `multiplier` (как сейчас);
- weight_diff: `delta`, `multiplier` (= strength).

Реализация может расширить `ActiveLoRALayer` union-полями или разделить на
два frozen dataclass’а; контракт forward — сумма дельт.

## Поток одного job

1. Handler валидирует schema + `normalize_lora_requests` (type + per-type strength).
2. Catalog resolve names → paths.
3. Loader читает только selected files; ветка по type; CPU tensors.
4. Prompt encode / TE offload — без изменений.
5. Activation: GPU transfer + attach к Linear targets.
6. Denoise: base + lora deltas + weight_diff deltas.
7. Deactivate + free GPU adapters.
8. VAE decode.
9. Response includes applied list with `type`.

Пустой `loras` — полный skip load/activate.

## Ошибки

Client-facing (no `refresh_worker`):

- unknown name / duplicate / >4 / bad strength shape;
- unknown `type`;
- strength out of range for that type;
- corrupt safetensors;
- keys/shapes incompatible with declared type;
- unknown Linear target / dimension mismatch.

Сообщения: имя LoRA + safe reason; no absolute volume paths.

CUDA OOM / unexpected: existing cleanup via activation `finally`, then handler
policy unchanged.

## Тесты (acceptance)

Unit, без реальных community-файлов в git — synthetic tensors в tmp:

1. **Normalize**
   - default type → `"lora"`;
   - explicit `weight_diff` ok;
   - unknown type fail;
   - strength `4.0` fail for `lora`, ok for `weight_diff`;
   - strength `5.0` ok for `weight_diff`, `5.1` fail;
   - response/as_dict always includes type.
2. **Load lora**
   - existing A/B cases still pass;
   - `.diff`-only file with `type=lora` → fail.
3. **Load weight_diff**
   - synthetic `[1, 12]` delta under
     `diffusion_model.txtfusion.projector.diff` maps to `txtfusion.projector`
     when model has that Linear;
   - wrong shape fail;
   - A/B keys with `type=weight_diff` → fail.
4. **Runtime math**
   - for a small Linear, `base + strength * linear(x, delta)` matches
     fused-weight reference within float tolerance;
   - mix one lora + one weight_diff on different targets both contribute.
5. **Lifecycle**
   - deactivate clears both kinds of active layers.

## Совместимость

| Клиент | Поведение |
|--------|-----------|
| Старый: только `{name, strength}` | type default `lora`, strength ≤2 — без изменений |
| Новый: `type=weight_diff`, strength 3–5 | supported |
| Новый: `type=lora`, strength >2 | rejected |

## Открытые решения (зафиксированы)

| Вопрос | Решение |
|--------|---------|
| Доставка файла | Volume + name; no inline payload |
| Дискриминатор | Explicit `type`, default `lora` |
| Auto-detect | Нет; validate content against type |
| Strength ceilings | Per-type: lora 2.0, weight_diff 5.0 |
| Echo type in response | Always |
| Max adapters | 4 total, mixed types |
| Repo storage of fedor file | No |

## Связанные документы

- `docs/superpowers/specs/2026-07-17-krea2-runtime-lora-design.md` — base
  runtime LoRA design (superseded only where this doc conflicts: strength
  ceiling is now type-dependent; adapter payload kinds include weight_diff).
- Reference format (external, not vendored): CliffNodes/fedor_bypass
  `build_fedor_bypass.py` → key
  `diffusion_model.txtfusion.projector.diff`, shape `[1, 12]`.
