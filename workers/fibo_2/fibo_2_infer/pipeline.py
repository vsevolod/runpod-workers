"""Local-only Fibo components, native BF16 inference and component CPU offload."""

import gc
from pathlib import Path

import torch
from diffusers import AutoencoderKLFlux2, BriaFibo2Pipeline, BriaFibo2Transformer2DModel
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.bria_fibo2.image_processor import BriaFibo2ImageProcessor
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from shared.api import GenerationResult, RequestError
from shared.megapixels import canvas_at_megapixels, resize_references
from shared.prefetch import loading_component, wait_for_component
from .config import CHECKPOINTS, offload_mode
from .request import serialize_vgl


class BudgetedImageProcessor(BriaFibo2ImageProcessor):
    """Use native crop/normalization at the shared API's reference sizes."""

    reference_megapixels = None

    def preprocess(self, image, *, height=None, width=None, mask=None,
                   area=1024 * 1024, text_encoder_image_size=768):
        images = image if isinstance(image, list) else [image]
        if self.reference_megapixels is None or len(images) == 1:
            return super().preprocess(image, height=height, width=width, mask=mask, area=area,
                                      text_encoder_image_size=text_encoder_image_size)
        references = resize_references(images, self.reference_megapixels,
                                       multiple=self.config.vae_scale_factor, max_side=4096)
        tensors, encoder_images = [], []
        for reference in references:
            encoded, viewed = super().preprocess(
                [reference], height=reference.height, width=reference.width,
                text_encoder_image_size=text_encoder_image_size,
            )
            tensors.extend(encoded)
            encoder_images.extend(viewed)
        return tensors, encoder_images


class Runtime:
    def __init__(self, pipe, model_dir, variant):
        self.pipe = pipe
        self.model_dir = Path(model_dir)
        self.variant = variant

    def run(self, norm, seed):
        width, height = norm.width, norm.height
        if norm.type == 'image_edit' and width is None:
            width, height = canvas_at_megapixels(norm.images[0], norm.reference_megapixels, multiple=16)
        if len(norm.images) == 1 and width * height > int(norm.reference_megapixels * 1_000_000):
            raise RequestError('For a single Fibo reference, the canvas must fit reference_megapixels')
        self.pipe.image_processor.reference_megapixels = norm.reference_megapixels
        try:
            output = self.pipe(
                prompt=serialize_vgl(norm.prompt), image=list(norm.images) if norm.images else None,
                negative_prompt=serialize_vgl(norm.negative_prompt) if norm.negative_prompt is not None else None,
                width=width, height=height, num_inference_steps=norm.num_inference_steps,
                guidance_scale=norm.guidance_scale, num_images_per_prompt=norm.num_images,
                generator=torch.Generator('cpu').manual_seed(seed),
                fonts=[str(self.model_dir / 'fonts' / name) for name in ('Aileron-Regular.otf', 'DejaVuSans.ttf')],
            )
        finally:
            self.pipe.maybe_free_model_hooks()
        repo, revision = CHECKPOINTS[self.variant]
        return GenerationResult(output.images, [], {'variant': self.variant},
                                {'model_id': repo, 'model_revision': revision,
                                 'model_variant': self.variant, 'scheduler': 'flow_match_euler'})

    def close(self):
        pipe, self.pipe = self.pipe, None
        if pipe is not None:
            pipe.remove_all_hooks()


def load_pipeline(model_dir, *, variant, device='cuda'):
    mode = offload_mode()
    root = Path(model_dir)
    transformer = text_encoder = processor = vae = scheduler = pipe = None
    try:
        wait_for_component('transformer')
        with loading_component('transformer'):
            transformer = BriaFibo2Transformer2DModel.from_pretrained(
                root / 'transformer', local_files_only=True, torch_dtype=torch.bfloat16).eval()
        wait_for_component('text_encoder')
        with loading_component('text_encoder'):
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                root / 'text_encoder', local_files_only=True, dtype=torch.bfloat16).eval()
            processor = Qwen3VLProcessor.from_pretrained(root / 'processor', local_files_only=True)
        wait_for_component('vae')
        with loading_component('vae'):
            vae = AutoencoderKLFlux2.from_pretrained(
                root / 'vae', local_files_only=True, torch_dtype=torch.bfloat16).eval()
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(root / 'scheduler', local_files_only=True)
        pipe = BriaFibo2Pipeline(transformer=transformer, text_encoder=text_encoder, processor=processor,
                                 vae=vae, scheduler=scheduler)
        pipe.image_processor = BudgetedImageProcessor(vae_scale_factor=pipe.vae_scale_factor * 2)
        if mode == 'cpu':
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        wait_for_component('all')
        return Runtime(pipe, root, variant)
    except BaseException:
        if pipe is not None:
            pipe.remove_all_hooks()
        pipe = transformer = text_encoder = processor = vae = scheduler = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        raise
