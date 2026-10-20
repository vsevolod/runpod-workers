"""Construct Gliese on the worker and load each prepared shard directly to GPU."""

import gc
import logging
import os
from pathlib import Path

import torch
from accelerate import init_empty_weights, load_checkpoint_in_model
from transformers import AutoConfig, AutoProcessor, GenerationConfig, Qwen3_5ForConditionalGeneration

from shared.prefetch import loading_component, wait_for_component
from model_store import MODEL_ID, MODEL_REVISION, shard_names

logger = logging.getLogger(__name__)


class ModelHandler:
    def __init__(self, root, *, device='cuda:0'):
        self.model = None
        self.processor = None
        self.model_id = MODEL_ID
        self.root = Path(root)
        try:
            self.load_models(device)
        except BaseException:
            self.close()
            raise

    def load_models(self, device):
        max_pixels = int(os.environ.get('CAPTION_MAX_PIXELS', '1048576'))
        if not 65536 <= max_pixels <= 4194304:
            raise ValueError('CAPTION_MAX_PIXELS must be in 65536..4194304')
        wait_for_component('processor')
        with loading_component('processor'):
            self.processor = AutoProcessor.from_pretrained(str(self.root), local_files_only=True)
            self.processor.image_processor.size = {'shortest_edge': 65536, 'longest_edge': max_pixels}
            config = AutoConfig.from_pretrained(str(self.root), local_files_only=True)
            if config.model_type != 'qwen3_5' or config.architectures != ['Qwen3_5ForConditionalGeneration']:
                raise ValueError('MODEL_DIR must contain the Gliese Qwen3.5 caption checkpoint')
            config._attn_implementation = 'sdpa'
            generation_config = GenerationConfig.from_pretrained(str(self.root), local_files_only=True)
        with loading_component('model_init'):
            # Allocate shapes only. The downloader never imports Torch; shards
            # are memory-mapped from local disk and moved directly to the device.
            with init_empty_weights():
                self.model = Qwen3_5ForConditionalGeneration(config)
        for index, name in enumerate(shard_names(self.root)):
            component = f'shard_{index:05}'
            wait_for_component(component)
            with loading_component(component):
                load_checkpoint_in_model(self.model, str(self.root / name),
                                         device_map={'': device}, dtype=torch.bfloat16, strict=False)
        self.model.tie_weights()
        missing = [name for name, value in self.model.named_parameters() if value.is_meta]
        if missing:
            raise ValueError(f'Gliese checkpoint left parameters unloaded: {missing[:5]}')
        self.model.to(device)
        self.model.generation_config = generation_config
        self.model.eval()
        # Later transfer errors must fail startup even after early shard loads.
        wait_for_component('all')
        logger.info('Gliese ready model=%s revision=%s max_pixels=%s', MODEL_ID, MODEL_REVISION, max_pixels)

    def close(self):
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
