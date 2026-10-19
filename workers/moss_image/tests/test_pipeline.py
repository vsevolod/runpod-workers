"""Execute the real MOSS sampling loop with small CPU components, without weights."""

from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path[:0] = [str(Path(__file__).parents[2]), str(Path(__file__).parents[1])]


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def pipeline(self):
        from moss_image_infer.vendor.pipeline_moss_image import MOSSImageTransformer, MOSSImagePipeline, MOSSImageScheduler
        from diffusers import AutoencoderKLFlux2
        transformer = MOSSImageTransformer(dim=16, ffn_dim=32, num_heads=2, num_layers=1,
                                           text_dim=8, freq_dim=8)
        vae = AutoencoderKLFlux2(block_out_channels=(16, 16, 16, 16), layers_per_block=1,
                                latent_channels=32, norm_num_groups=8)
        pipe = MOSSImagePipeline(transformer, MOSSImageScheduler(), vae=vae)
        # Tokenization/9B encoder are the expensive boundary; denoising/VAE are real.
        def encode(prompt, max_sequence_length=4096):
            return torch.ones(1, 2, 8), torch.ones(1, 2, dtype=torch.bool)
        pipe.encode_prompt = encode
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def norm(self, variant):
        from shared.api import MOSS_API, parse_job_input
        from moss_image_infer.request import normalize_job_input
        parsed = parse_job_input({'input': {'prompt': 'fox', 'width': 256, 'height': 256,
                                            'num_inference_steps': 2, 'seed': 0,
                                            'model_options': {'variant': variant, 'uncond_skip_layers': []}}}, MOSS_API)
        return normalize_job_input(parsed.values, raw_keys=parsed.raw_keys)

    def test_base_turbo_base_decodes_and_restores_same_seed_output(self):
        from moss_image_infer.pipeline import Runtime
        from safetensors.torch import save_file
        pipe = self.pipeline()
        # Exercise the actual Accelerate hook lifecycle on CPU; GPU memory fit
        # remains a separate real-model check.
        pipe.enable_model_cpu_offload(device='cpu')
        module = pipe.transformer.layers[0].proj['q']
        state = {'diffusion_model.layers.0.proj.q.lora_down.weight': torch.ones(1, 16),
                 'diffusion_model.layers.0.proj.q.lora_up.weight': torch.ones(16, 1) * .05,
                 'diffusion_model.layers.0.proj.q.alpha': torch.tensor(1.)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'lora.safetensors'
            save_file(state, str(path))
            runtime = Runtime(pipe, Path(tmp))
            with patch('moss_image_infer.pipeline.ensure_turbo_lora', return_value=path):
                first = runtime.run(self.norm('base'), 0)
                fast = runtime.run(self.norm('turbo'), 0)
                last = runtime.run(self.norm('base'), 0)
            self.assertEqual(first.images[0].size, (256, 256))
            self.assertEqual(fast.images[0].size, (256, 256))
            self.assertEqual(first.images[0].tobytes(), last.images[0].tobytes())
            self.assertIs(pipe.transformer.layers[0].proj['q'], module)
            self.assertEqual(first.loras, [])
            self.assertEqual(fast.loras[0].as_dict()['strength'], .7)
            runtime.close()
            runtime.close()
            self.assertIsNone(runtime.pipe)

    def test_failure_offloads_components_and_restores_transformer(self):
        from moss_image_infer.pipeline import Runtime
        pipe = self.pipeline()
        runtime = Runtime(pipe, Path('/unused'))
        with patch.object(pipe.transformer, 'forward', side_effect=RuntimeError('failed')), patch.object(
            pipe, 'maybe_free_model_hooks', wraps=pipe.maybe_free_model_hooks
        ) as cleanup:
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                runtime.run(self.norm('base'), 0)
            cleanup.assert_called()

    def test_loads_separate_local_components_without_remote_code(self):
        from moss_image_infer.pipeline import load_pipeline
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5Model
        pipe = self.pipeline()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'MOSS_OFFLOAD': 'none'}):
            root = Path(tmp)
            pipe.transformer.save_pretrained(root / 'transformer')
            pipe.vae.save_pretrained(root / 'vae')
            pipe.scheduler.save_pretrained(root / 'scheduler')
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'[UNK]': 0, 'fox': 1}, unk_token='[UNK]')),
                                                unk_token='[UNK]')
            tokenizer.chat_template = '[UNK] {{ messages[0]["content"] }}'
            tokenizer.save_pretrained(root / 'tokenizer')
            config = Qwen3_5Config(
                text_config={'vocab_size': 8, 'hidden_size': 8, 'intermediate_size': 16, 'num_hidden_layers': 2,
                             'num_attention_heads': 1, 'num_key_value_heads': 1, 'head_dim': 8,
                             'linear_key_head_dim': 4, 'linear_value_head_dim': 4,
                             'linear_num_key_heads': 2, 'linear_num_value_heads': 2, 'linear_conv_kernel_dim': 2,
                             'layer_types': ['linear_attention', 'full_attention'],
                             'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000.,
                                                 'partial_rotary_factor': 1., 'mrope_section': [1, 1, 2]}},
                vision_config={'hidden_size': 8, 'intermediate_size': 16, 'depth': 1, 'num_heads': 1,
                               'out_hidden_size': 8, 'num_position_embeddings': 16, 'patch_size': 2,
                               'spatial_merge_size': 2, 'temporal_patch_size': 1},
                image_token_id=4, video_token_id=5, vision_start_token_id=6, vision_end_token_id=7,
            )
            Qwen3_5Model(config).save_pretrained(root / 'text_encoder')
            runtime = load_pipeline(root, device='cpu')
            result = runtime.run(self.norm('base'), 0)
            self.assertEqual(result.images[0].size, (256, 256))
            runtime.close()
