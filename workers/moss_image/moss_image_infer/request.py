"""MOSS settings; base denotes SFT, with its Edit adapter for image_edit."""

from dataclasses import dataclass

from shared.api import CommonRequest, MOSS_API, RequestError, normalize_common_fields


@dataclass(frozen=True)
class NormalizedRequest(CommonRequest):
    model_variant: str
    width: int
    height: int
    shift: float
    uncond_skip_layers: tuple[int, ...]


def normalize_job_input(values, *, raw_keys):
    editing = values['type'] == 'image_edit'
    if values.get('loras'):
        raise RequestError('MOSS supports only its built-in Turbo and Edit adapters')
    options = values['model_options']
    variant = options['variant'] or ('base' if editing else 'turbo')
    if editing and variant == 'turbo':
        raise RequestError('MOSS image_edit requires variant=base; Turbo editing is not supported')
    values = dict(values)
    if 'guidance_scale' not in raw_keys:
        values['guidance_scale'] = 4. if editing else (1. if variant == 'turbo' else 3.)
    steps = 30 if editing else (8 if variant == 'turbo' else 50)
    common = normalize_common_fields(values, MOSS_API, raw_keys=raw_keys,
                                     default_steps=steps,
                                     megapixel_model='moss_image.edit' if editing else f'moss_image.{variant}')
    if editing and int(common.reference_megapixels * 1_000_000) < len(common.images) * 16**2:
        raise RequestError('reference_megapixels is too small for the selected references')
    skip = options['uncond_skip_layers']
    skip = tuple(skip) if skip is not None else (() if variant == 'turbo' else (22,))
    if variant == 'turbo' and (common.guidance_scale != 1. or common.negative_prompt or skip):
        raise RequestError('MOSS Turbo requires guidance_scale=1, no negative prompt and no layer skipping')
    width, height = values['width'], values['height']
    if width * height > 2048 * 2048:
        raise RequestError('MOSS width*height must be <= 4194304 pixels')
    return NormalizedRequest(**vars(common), model_variant=variant, width=width, height=height,
                             shift=options['shift'], uncond_skip_layers=skip)
