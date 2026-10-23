#!/usr/bin/env python3
"""Generate/edit with MOSS-Image-SFT on the common RunPod endpoint, or fetch a job.

Examples:
  python scripts/moss_image.py --prompt 'A red fox' -o moss.png
  python scripts/moss_image.py --variant base --prompt 'A poster reading "HELLO"'
  python scripts/moss_image.py -i source.png --prompt 'Make the jacket blue' -o edit.png
  python scripts/moss_image.py --request-id JOB_ID -o moss.png

Environment: RUNPOD_API_KEY, optional ENDPOINT_ID override.
"""

import argparse
import os
from pathlib import Path
import sys

# Reuse the existing stdlib-only HTTP/polling/image-delivery client helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_image_21 import (
    DEFAULT_ENDPOINT_ID, _require_env, _prompt_line, _submit_job,
    _poll_job, _save_result, _run_fetch, _step_count,
    _image_to_data_url,
)


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prompt')
    parser.add_argument('--variant', choices=('base', 'turbo'), default=None,
                        help='Generation defaults to Turbo; edits use base + Edit LoRA (30 steps, CFG 4)')
    parser.add_argument('-i', '--image', action='append', default=[], help='Edit reference image; repeat up to four times')
    parser.add_argument('--reference-megapixels', type=lambda v: 'max' if v == 'max' else float(v), default='max')
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--steps', type=_step_count)
    parser.add_argument('--cfg', '--guidance-scale', dest='guidance_scale', type=float)
    parser.add_argument('--negative-prompt')
    parser.add_argument('--shift', type=float)
    parser.add_argument('--skip-layers', help='Base negative-branch layers, e.g. 22; empty disables skipping')
    parser.add_argument('--num-images', type=int, default=1)
    parser.add_argument('--request-id')
    parser.add_argument('--poll-interval', type=float, default=5.)
    parser.add_argument('-o', '--output')
    return parser


def _run_generate(args, *, endpoint_id, api_key):
    prompt = args.prompt if args.prompt is not None else _prompt_line('prompt')
    if not prompt.strip():
        print('error: prompt is required', file=sys.stderr)
        return 1
    editing = bool(args.image)
    variant = args.variant or ('base' if editing else 'turbo')
    if len(args.image) > 4 or editing and (variant == 'turbo' or args.num_images != 1):
        print('error: MOSS Edit requires 1..4 references, variant=base and num-images=1', file=sys.stderr)
        return 1
    options = {'variant': variant}
    if args.shift is not None:
        options['shift'] = args.shift
    if args.skip_layers is not None:
        try:
            options['uncond_skip_layers'] = [int(i.strip()) for i in args.skip_layers.split(',') if i.strip()]
        except ValueError:
            print('error: --skip-layers must be comma-separated integers', file=sys.stderr)
            return 1
    values = {'model': 'moss_image', 'type': 'image_edit' if editing else 'image_generate', 'prompt': prompt,
              'width': args.width, 'height': args.height, 'num_images': args.num_images,
              'model_options': options}
    if editing:
        values['images'] = [_image_to_data_url(Path(path)) for path in args.image]
        values['reference_megapixels'] = args.reference_megapixels
    for name, value in (('seed', args.seed), ('num_inference_steps', args.steps),
                        ('guidance_scale', args.guidance_scale), ('negative_prompt', args.negative_prompt)):
        if value is not None:
            values[name] = value
    job = _submit_job(endpoint_id, api_key, values)
    result = _poll_job(endpoint_id, api_key, job, interval=max(.5, args.poll_interval))
    return _save_result(result, Path(args.output or 'output.png'))


def main():
    args = _build_parser().parse_args()
    endpoint = os.environ.get('ENDPOINT_ID', '').strip() or DEFAULT_ENDPOINT_ID
    key = _require_env('RUNPOD_API_KEY')
    if args.request_id is not None:
        return _run_fetch(args, endpoint_id=endpoint, api_key=key)
    return _run_generate(args, endpoint_id=endpoint, api_key=key)


if __name__ == '__main__':
    raise SystemExit(main())
