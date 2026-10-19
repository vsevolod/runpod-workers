"""Pinned MOSS pipeline with component offload and request-scoped Turbo LoRA."""

from contextlib import nullcontext
from dataclasses import dataclass
import logging
import os
from pathlib import Path

import torch
from diffusers import AutoencoderKLFlux2
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from shared.api import GenerationResult
from .config import MODEL_ID, MODEL_REVISION, LORA_ID, LORA_REVISION, LORA_STRENGTH, offload_mode
from .lora import activate_lora
from .model_store import ensure_turbo_lora
from .vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurboSelection:
    def as_dict(self):
        return {'name': 'moss_sft_dmd_8_steps', 'strength': LORA_STRENGTH, 'type': 'lora'}


class Runtime:
    def __init__(self, pipe, model_dir):
        self.pipe = pipe
        self.model_dir = Path(model_dir)
        self._turbo_state = None

    def run(self, norm, seed):
        turbo = norm.model_variant == 'turbo'
        if turbo and self._turbo_state is None:
            path = ensure_turbo_lora(self.model_dir, hf_home=os.environ.get('HF_HOME'),
                                    token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'))
            state = load_file(str(path), device='cpu')
            self._turbo_state = {key: value.to(torch.bfloat16) if key.endswith('.weight') else value
                                 for key, value in state.items()}
        context = activate_lora(self.pipe.transformer, self._turbo_state, strength=LORA_STRENGTH) if turbo else nullcontext()
        self.pipe.scheduler = MOSSImageScheduler(
            num_train_timesteps=self.pipe.scheduler.config.num_train_timesteps,
            training_shift=self.pipe.scheduler.config.training_shift, shift=norm.shift,
        )
        logger.info('MOSS variant=%s steps=%s cfg=%s shift=%s lora=%s strength=%s',
                    norm.model_variant, norm.num_inference_steps, norm.guidance_scale, norm.shift,
                    LORA_ID if turbo else 'none', LORA_STRENGTH if turbo else 0.)
        try:
            with context:
                output = self.pipe(
                    norm.prompt, width=norm.width, height=norm.height,
                    num_inference_steps=norm.num_inference_steps, guidance_scale=norm.guidance_scale,
                    negative_prompt=norm.negative_prompt or '', num_images_per_prompt=norm.num_images,
                    uncond_skip_layers=norm.uncond_skip_layers,
                    generator=torch.Generator('cpu').manual_seed(seed),
                )
        finally:
            # Also release offload hooks when encoding/denoising/decoding raises.
            self.pipe.maybe_free_model_hooks()
        return GenerationResult(
            output.images, [TurboSelection()] if turbo else [],
            {'variant': norm.model_variant, 'shift': norm.shift, 'uncond_skip_layers': list(norm.uncond_skip_layers)},
            {'model_id': MODEL_ID, 'model_revision': MODEL_REVISION, 'model_variant': norm.model_variant,
             'scheduler': 'moss_image', 'lora_revision': LORA_REVISION if turbo else None},
        )

    def close(self):
        pipe, self.pipe = self.pipe, None
        self._turbo_state = None
        if pipe is not None:
            pipe.remove_all_hooks()


def load_pipeline(model_dir, *, device='cuda'):
    mode = offload_mode()
    root = Path(model_dir)
    pipe = MOSSImagePipeline(
        transformer=MOSSImageTransformer.from_pretrained(
            root / 'transformer', local_files_only=True, torch_dtype=torch.bfloat16).eval(),
        text_encoder=AutoModel.from_pretrained(
            root / 'text_encoder', local_files_only=True, dtype=torch.bfloat16).eval(),
        tokenizer=AutoTokenizer.from_pretrained(root / 'tokenizer', local_files_only=True),
        vae=AutoencoderKLFlux2.from_pretrained(
            root / 'vae', local_files_only=True, torch_dtype=torch.float32).eval(),
        scheduler=MOSSImageScheduler.from_pretrained(root / 'scheduler', local_files_only=True),
    )
    if mode == 'cpu':
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return Runtime(pipe, root)
