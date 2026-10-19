"""Generation-only MOSS settings; base denotes SFT without the DMD adapter."""

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
    if values['type'] != 'image_generate':
        raise RequestError('MOSS supports only image_generate')
    if values.get('loras'):
        raise RequestError('MOSS currently supports only its built-in Turbo LoRA')
    options = values['model_options']
    variant = options['variant']
    values = dict(values)
    if 'guidance_scale' not in raw_keys:
        values['guidance_scale'] = 1. if variant == 'turbo' else 3.
    common = normalize_common_fields(values, MOSS_API, raw_keys=raw_keys)
    skip = options['uncond_skip_layers']
    skip = tuple(skip) if skip is not None else (() if variant == 'turbo' else (22,))
    if variant == 'turbo' and (common.guidance_scale != 1. or common.negative_prompt or skip):
        raise RequestError('MOSS Turbo requires guidance_scale=1, no negative prompt and no layer skipping')
    width, height = values['width'], values['height']
    if width * height > 2048 * 2048:
        raise RequestError('MOSS width*height must be <= 4194304 pixels')
    return NormalizedRequest(**vars(common), model_variant=variant, width=width, height=height,
                             shift=options['shift'], uncond_skip_layers=skip)
