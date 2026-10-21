"""VGL objects and base/turbo capabilities, validated before selecting weights."""

from dataclasses import dataclass
import json

from shared.api import CommonRequest, FIBO_API, RequestError, normalize_common_fields
from shared.megapixels import canvas_at_megapixels


def serialize_vgl(value, *, field='prompt'):
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    except (ValueError, TypeError, RecursionError) as err:
        raise RequestError(f'{field} must contain valid finite JSON values: {err}') from err


def _validate_scene(prompt, *, editing):
    serialize_vgl(prompt)
    if editing:
        if 'structured_caption' in prompt and set(prompt) - {'structured_caption', 'edit_instruction'}:
            raise RequestError('A nested edit prompt may contain only structured_caption and edit_instruction')
        instruction = prompt.get('edit_instruction')
        if not isinstance(instruction, str) or not instruction.strip():
            raise RequestError('image_edit requires a non-empty prompt.edit_instruction')
    elif 'edit_instruction' in prompt:
        raise RequestError('prompt.edit_instruction is only valid for image_edit')
    caption = prompt.get('structured_caption', {k: v for k, v in prompt.items() if k != 'edit_instruction'})
    if not isinstance(caption, dict):
        raise RequestError('prompt.structured_caption must be a JSON object')
    if caption or not editing:
        description = caption.get('short_description')
        if not isinstance(description, str) or not description.strip():
            raise RequestError('VGL caption requires a non-empty short_description')


@dataclass(frozen=True)
class NormalizedRequest(CommonRequest):
    model_variant: str
    width: int | None
    height: int | None


def normalize_job_input(values, *, raw_keys):
    variant = values['model_options']['variant']
    values = dict(values)
    if 'guidance_scale' not in raw_keys:
        values['guidance_scale'] = 1. if variant == 'turbo' else 5.
    common = normalize_common_fields(values, FIBO_API, raw_keys=raw_keys,
                                     megapixel_model=f'fibo_2.{variant}')
    editing = common.type == 'image_edit'
    _validate_scene(common.prompt, editing=editing)
    if common.negative_prompt is not None:
        serialize_vgl(common.negative_prompt, field='negative_prompt')
    if values['loras']:
        raise RequestError('Fibo 2 does not support additional LoRAs')
    if variant == 'turbo':
        if not editing:
            raise RequestError('Fibo turbo supports only image_edit; use base for generation')
        if common.guidance_scale != 1. or common.negative_prompt is not None:
            raise RequestError('Fibo turbo requires guidance_scale=1 and no negative_prompt')
    if ('width' in raw_keys) != ('height' in raw_keys):
        raise RequestError('width and height must be supplied together')
    width, height = values['width'], values['height']
    if width * height > 4194304:
        raise RequestError('Fibo width*height must be <= 4194304 pixels')
    if editing and 'width' not in raw_keys:
        width = height = None  # Preserve the native source-aspect canvas selection.
    if editing:
        budget = int(common.reference_megapixels * 1_000_000)
        if budget < len(common.images) * 16**2:
            raise RequestError('reference_megapixels is too small for the selected references')
        canvas = (width, height) if width is not None else canvas_at_megapixels(
            common.images[0], common.reference_megapixels, multiple=16)
        if max(canvas) > 4096 or canvas[0] * canvas[1] > 4194304:
            raise RequestError('Automatic edit canvas exceeds Fibo limits; supply width and height')
        if len(common.images) == 1 and canvas[0] * canvas[1] > budget:
            raise RequestError('For a single Fibo reference, the canvas must fit reference_megapixels')
    return NormalizedRequest(**vars(common), model_variant=variant, width=width, height=height)
