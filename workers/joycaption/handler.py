"""RunPod image-to-caption API backed by Gliese Qwen3.5 9B.

Portions of the RunPod handler pattern are adapted from runpod-workers/worker-sdxl
under the MIT License; see LICENSES/RUNPOD-WORKER-SDXL-MIT.txt.
"""

import logging
import os
from pathlib import Path
import sys
import time

workers = Path(__file__).resolve().parents[1]
if (workers / 'shared').is_dir():
    sys.path.insert(0, str(workers))

import runpod
from runpod.serverless.utils.rp_validator import validate

from caption_utils import ImageError, decode_image_base64, parse_max_image_pixels, resolve_prompt
from model_store import MODEL_REVISION, model_directory, prefetch_model
from schemas import INPUT_SCHEMA

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'),
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('caption.handler')
MODELS = None


def caption_image(image, prompt, *, max_new_tokens=512, temperature=0.6, top_p=0.9):
    import torch

    processor = MODELS.processor
    messages = [{'role': 'user', 'content': [
        {'type': 'image'}, {'type': 'text', 'text': prompt},
    ]}]
    # Use the checkpoint's caption template and explicit non-thinking mode.
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[text], images=[image], return_tensors='pt').to(MODELS.model.device)
    kwargs = dict(max_new_tokens=int(max_new_tokens), do_sample=temperature > 0, use_cache=True)
    if temperature > 0:
        kwargs.update(temperature=float(temperature), top_p=float(top_p), top_k=None)
    with torch.inference_mode():
        outputs = MODELS.model.generate(**inputs, **kwargs)
    generated = [outputs[0][inputs['input_ids'].shape[-1]:]]
    return processor.batch_decode(generated, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0].strip()


def handler(job):
    import torch

    try:
        job_input = job['input']
    except (KeyError, TypeError):
        return {'error': "Job must contain an 'input' object"}
    validated = validate(job_input, INPUT_SCHEMA)
    if 'errors' in validated:
        return {'error': validated['errors']}
    data = validated['validated_input']
    try:
        image = decode_image_base64(data['image'], max_pixels=parse_max_image_pixels())
    except (ImageError, ValueError) as err:
        return {'error': str(err)}
    prompt = resolve_prompt(data.get('prompt'))
    started = time.monotonic()
    logger.info('caption begin job=%s image=%sx%s max_new_tokens=%s',
                job.get('id', 'local'), image.width, image.height, data['max_new_tokens'])
    try:
        caption = caption_image(image, prompt, max_new_tokens=int(data['max_new_tokens']),
                                temperature=float(data['temperature']), top_p=float(data['top_p']))
        if not caption:
            raise RuntimeError('Caption model returned an empty answer')
    except torch.cuda.OutOfMemoryError:
        logger.exception('CUDA OOM during captioning')
        torch.cuda.empty_cache()
        return {'error': 'CUDA out of memory. Reduce CAPTION_MAX_PIXELS or use a larger GPU.',
                'refresh_worker': True}
    except Exception as err:
        logger.exception('Captioning failed')
        return {'error': f'{type(err).__name__}: {err}', 'refresh_worker': True}
    logger.info('caption done job=%s elapsed_s=%.2f chars=%s',
                job.get('id', 'local'), time.monotonic() - started, len(caption))
    return {'caption': caption, 'prompt': prompt, 'model': MODELS.model_id,
            'model_revision': MODEL_REVISION}


def main():
    global MODELS
    root = model_directory()
    # Start the CPU downloader before importing Torch/Transformers/Accelerate.
    with prefetch_model(root):
        from caption_runtime import ModelHandler
        MODELS = ModelHandler(root)
    runpod.serverless.start({'handler': handler})


if __name__ == '__main__':
    main()
