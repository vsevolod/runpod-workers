"""Actual tiny local weights with a controlled CPU-only downloader process."""
import gc
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch
import weakref

sys.path[:0] = [str(Path(__file__).parents[2]), str(Path(__file__).parents[1])]


class FiboLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from diffusers import AutoencoderKLFlux2, BriaFibo2Transformer2DModel, FlowMatchEulerDiscreteScheduler
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
        cls.temp = tempfile.TemporaryDirectory()
        cls.source = Path(cls.temp.name)
        transformer = BriaFibo2Transformer2DModel(
            in_channels=1, patch_size=2, dim=32, n_layers=2, n_refiner_layers=1, n_heads=2,
            cap_feat_dim=48, injection_layer_ids=(1,), perceiver_num_layers=1,
            min_num_gist_tokens=4, max_num_gist_tokens=8, gist_step=4,
            gist_min_text_len=4, gist_max_text_len=16, axes_dims=(4, 6, 6), axes_lens=(16, 16, 16))
        transformer.save_pretrained(cls.source / 'transformer')
        config = Qwen3VLConfig(
            text_config={'vocab_size': 64, 'hidden_size': 16, 'intermediate_size': 32,
                         'num_hidden_layers': 2, 'num_attention_heads': 2,
                         'num_key_value_heads': 2, 'head_dim': 8},
            vision_config={'depth': 1, 'hidden_size': 16, 'intermediate_size': 32,
                           'num_heads': 2, 'out_hidden_size': 16, 'patch_size': 2,
                           'spatial_merge_size': 2, 'temporal_patch_size': 2,
                           'deepstack_visual_indexes': []})
        Qwen3VLForConditionalGeneration(config).save_pretrained(cls.source / 'text_encoder')
        AutoencoderKLFlux2(sample_size=32, in_channels=3, out_channels=3,
            down_block_types=('DownEncoderBlock2D',), up_block_types=('UpDecoderBlock2D',),
            block_out_channels=(4,), layers_per_block=1, latent_channels=1, norm_num_groups=1,
            use_quant_conv=False, use_post_quant_conv=False).save_pretrained(cls.source / 'vae')
        FlowMatchEulerDiscreteScheduler().save_pretrained(cls.source / 'scheduler')
        cls.addClassCleanup(cls.temp.cleanup)

    def exercise(self, mode):
        from fibo_2_infer import pipeline
        from shared.prefetch import PrefetchSession, DownloadError
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            os.mkfifo(root / 'gate')
            child = root / 'download.py'
            child.write_text(textwrap.dedent('''
                import argparse, shutil, sys
                from pathlib import Path
                from shared.prefetch import ComponentPublisher
                p = argparse.ArgumentParser()
                p.add_argument('--prefetch-state-dir')
                p.add_argument('--source'); p.add_argument('--mode')
                a = p.parse_args()
                assert 'torch' not in sys.modules
                root = Path(__file__).parent
                publisher = ComponentPublisher(a.prefetch_state_dir)
                shutil.copytree(Path(a.source) / 'transformer', root / 'transformer')
                publisher.ready('transformer', root / 'transformer')
                with (root / 'gate').open() as gate: gate.read()
                if a.mode == 'late_failure': raise SystemExit(7)
                for name in ('text_encoder', 'vae', 'scheduler'):
                    shutil.copytree(Path(a.source) / name, root / name)
                    if name == 'text_encoder': publisher.ready(name, root / name)
                publisher.ready('vae', root / 'vae')
                publisher.ready('all', root)
                if a.mode == 'bad_exit': raise SystemExit(9)
            '''))
            session = PrefetchSession([sys.executable, '-u', str(child), '--source', str(self.source), '--mode', mode],
                                      env={**os.environ, 'PYTHONPATH': str(Path(__file__).parents[2])})
            original = pipeline.BriaFibo2Transformer2DModel.from_pretrained
            loaded = []
            def consume(*args, **kwargs):
                model = original(*args, **kwargs)
                loaded.append(weakref.ref(model))
                self.assertFalse((root / 'text_encoder').exists())
                self.assertIsNone(session.process.poll())
                if mode == 'loading_failure':
                    del model
                    raise RuntimeError('load failed after weights were read')
                with (root / 'gate').open('w') as gate: gate.write('loaded')
                return model
            with patch.object(pipeline.BriaFibo2Transformer2DModel, 'from_pretrained', side_effect=consume), patch.object(
                pipeline.Qwen3VLProcessor, 'from_pretrained', return_value=None
            ), patch.dict(os.environ, {'FIBO_OFFLOAD': 'none'}):
                if mode == 'success':
                    with session:
                        runtime = pipeline.load_pipeline(root, variant='base', device='cpu')
                    self.assertEqual(runtime.pipe.transformer.dtype, pipeline.torch.bfloat16)
                    self.assertEqual(runtime.pipe.vae.dtype, pipeline.torch.bfloat16)
                    self.assertEqual(runtime.pipe.text_encoder.dtype, pipeline.torch.bfloat16)
                    runtime.close()
                else:
                    error = RuntimeError if mode == 'loading_failure' else DownloadError
                    with self.assertRaises(error):
                        with session:
                            pipeline.load_pipeline(root, variant='turbo', device='cpu')
            gc.collect()
            self.assertIsNotNone(session.process.poll())
            self.assertFalse(session.state_dir.exists())
            self.assertTrue(loaded)
            self.assertTrue(all(ref() is None for ref in loaded), 'Partially loaded weights remain owned')

    def test_loading_overlaps_later_downloads_using_actual_bf16_weights(self):
        self.exercise('success')

    def test_later_download_failure_and_nonzero_final_exit_release_weights(self):
        for mode in ('late_failure', 'bad_exit'):
            with self.subTest(mode=mode): self.exercise(mode)

    def test_weight_loading_failure_terminates_blocked_child(self):
        self.exercise('loading_failure')
