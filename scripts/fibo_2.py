#!/usr/bin/env python3
"""Generate/edit with Fibo-2 VGL JSON on the common RunPod image endpoint.

  python scripts/fibo_2.py --prompt-file scene.json -o output.png
  python scripts/fibo_2.py --prompt-file edit.json --image source.png --variant turbo
  python scripts/fibo_2.py --request-id JOB_ID -o output.png

Environment: RUNPOD_API_KEY, optional ENDPOINT_ID override.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_image_21 import (
    DEFAULT_ENDPOINT_ID, _require_env, _image_to_data_url, _submit_job,
    _poll_job, _save_result, _run_fetch, _step_count,
)


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prompt-file', type=Path, help='VGL JSON object, with edit_instruction for editing')
    parser.add_argument('--negative-prompt-file', type=Path, help='Negative VGL JSON object (base only)')
    parser.add_argument('--image', type=Path, action='append', default=[], help='Ordered edit reference; repeat up to four')
    parser.add_argument('--variant', choices=('base', 'turbo'), default='base')
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--reference-megapixels', type=lambda value: 'max' if value == 'max' else float(value), default='max')
    parser.add_argument('--steps', type=_step_count)
    parser.add_argument('--cfg', '--guidance-scale', dest='guidance_scale', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num-images', type=int, default=1)
    parser.add_argument('--request-id')
    parser.add_argument('--poll-interval', type=float, default=5.)
    parser.add_argument('-o', '--output')
    return parser


def _read_object(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or not value:
        raise ValueError(f'{path} must contain a non-empty JSON object')
    return value


def _run_generate(args, *, endpoint_id, api_key):
    if args.prompt_file is None:
        print('error: --prompt-file is required', file=sys.stderr)
        return 1
    try:
        values = {'model': 'fibo_2', 'type': 'image_edit' if args.image else 'image_generate',
                  'prompt': _read_object(args.prompt_file), 'model_options': {'variant': args.variant},
                  'num_images': args.num_images, 'reference_megapixels': args.reference_megapixels}
        if args.negative_prompt_file is not None:
            values['negative_prompt'] = _read_object(args.negative_prompt_file)
    except (OSError, ValueError) as err:
        print(f'error: {err}', file=sys.stderr)
        return 1
    if args.image:
        values['images'] = [_image_to_data_url(path) for path in args.image]
    for name, value in (('width', args.width), ('height', args.height), ('seed', args.seed),
                        ('num_inference_steps', args.steps), ('guidance_scale', args.guidance_scale)):
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
