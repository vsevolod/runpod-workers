"""Actual tiny Qwen3.5 weights, coordinated with a controlled CPU subprocess."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from safetensors.torch import save_file
from transformers import GenerationConfig, Qwen3_5Config, Qwen3_5ForConditionalGeneration
from shared.prefetch import DownloadError, PrefetchSession
import caption_runtime


def checkpoint(root):
    config = Qwen3_5Config(
        text_config={'vocab_size': 64, 'hidden_size': 64, 'intermediate_size': 128,
                     'num_hidden_layers': 4, 'num_attention_heads': 4, 'num_key_value_heads': 2,
                     'head_dim': 16, 'layer_types': ['linear_attention'] * 3 + ['full_attention'],
                     'linear_num_key_heads': 2, 'linear_num_value_heads': 2,
                     'linear_key_head_dim': 8, 'linear_value_head_dim': 8, 'max_position_embeddings': 512,
                     'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000.,
                                         'partial_rotary_factor': .5, 'mrope_section': [1, 1, 2]}},
        vision_config={'hidden_size': 32, 'out_hidden_size': 64, 'intermediate_size': 64,
                       'depth': 1, 'num_heads': 4, 'deepstack_visual_indexes': [],
                       'num_position_embeddings': 16, 'patch_size': 16},
        image_token_id=60, vision_start_token_id=61, vision_end_token_id=62,
    )
    model = Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16).eval()
    config.architectures = ['Qwen3_5ForConditionalGeneration']
    config.save_pretrained(root)
    GenerationConfig.from_model_config(config).save_pretrained(root)
    state = {name: value.contiguous() for name, value in model.state_dict().items()}
    names = list(state)
    middle = len(names) // 2
    first, second = 'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'
    save_file({name: state[name] for name in names[:middle]}, root / first, metadata={'format': 'pt'})
    save_file({name: state[name] for name in names[middle:]}, root / 'second.pending', metadata={'format': 'pt'})
    (root / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {
        'total_size': sum(value.numel() * value.element_size() for value in state.values())}, 'weight_map': {
        name: first if i < middle else second for i, name in enumerate(names)}}))
    return model


def downloader(root, *, fail_late=False):
    os.mkfifo(root / 'gate')
    script = root / 'controlled.py'
    script.write_text(textwrap.dedent('''
        import argparse
        from pathlib import Path
        from shared.prefetch import ComponentPublisher
        parser = argparse.ArgumentParser()
        parser.add_argument('--prefetch-state-dir')
        args = parser.parse_args()
        root = Path(__file__).parent
        publisher = ComponentPublisher(args.prefetch_state_dir)
        publisher.ready('processor', root)
        publisher.ready('shard_00000', root / 'model-00001-of-00002.safetensors')
        with (root / 'gate').open('rb', buffering=0) as gate:
            assert gate.read(1) == b'1'
    ''') + ('raise SystemExit(7)\n' if fail_late is True else textwrap.dedent('''
        (root / 'second.pending').rename(root / 'model-00002-of-00002.safetensors')
        publisher.ready('shard_00001', root / 'model-00002-of-00002.safetensors')
        publisher.ready('all', root)
    ''')) + ('raise SystemExit(9)\n' if fail_late == 'after_all' else ''))
    workers = Path(__file__).parents[2]
    return PrefetchSession([sys.executable, '-u', str(script)],
                           env={**os.environ, 'PYTHONPATH': str(workers)})


@pytest.mark.parametrize('fail_late', [False, True, 'after_all'])
def test_first_shard_loads_before_later_download_and_later_failure_is_fatal(tmp_path, fail_late):
    original = checkpoint(tmp_path)
    session = downloader(tmp_path, fail_late=fail_late)
    actual_load = caption_runtime.load_checkpoint_in_model
    loaded = []

    def load(model, filename, **kwargs):
        actual_load(model, filename, **kwargs)
        loaded.append(Path(filename).name)
        if len(loaded) == 1:
            assert not (tmp_path / 'model-00002-of-00002.safetensors').exists()
            assert session.process.poll() is None
            with (tmp_path / 'gate').open('wb', buffering=0) as gate:
                gate.write(b'1')

    processor = SimpleNamespace(image_processor=SimpleNamespace(size={}))
    with patch.object(caption_runtime.AutoProcessor, 'from_pretrained', return_value=processor), \
         patch.object(caption_runtime, 'load_checkpoint_in_model', side_effect=load):
        if fail_late:
            with pytest.raises(DownloadError, match='9' if fail_late == 'after_all' else '7'):
                with session:
                    caption_runtime.ModelHandler(tmp_path, device='cpu')
        else:
            with session:
                runtime = caption_runtime.ModelHandler(tmp_path, device='cpu')
            assert len(loaded) == 2
            for name, tensor in original.state_dict().items():
                assert torch.equal(runtime.model.state_dict()[name], tensor), name
            inputs = torch.tensor([[1, 2, 3]])
            # Compare with Transformers' ordinary loader, including its buffer
            # initialization and recurrent-attention path, not just file bytes.
            reference = Qwen3_5ForConditionalGeneration.from_pretrained(
                tmp_path, local_files_only=True, dtype=torch.bfloat16,
                device_map={'': 'cpu'}, attn_implementation='sdpa').eval()
            with torch.inference_mode():
                assert torch.equal(runtime.model(inputs).logits, reference(inputs).logits)
                expected = reference.generate(inputs, max_new_tokens=2, do_sample=False)
                actual = runtime.model.generate(inputs, max_new_tokens=2, do_sample=False)
            assert torch.equal(actual, expected)
            assert runtime.processor.image_processor.size['longest_edge'] == 1048576
            runtime.close()
            assert runtime.model is None
    assert session.process.poll() is not None


def test_loading_failure_terminates_blocked_downloader(tmp_path):
    checkpoint(tmp_path)
    session = downloader(tmp_path)
    processor = SimpleNamespace(image_processor=SimpleNamespace(size={}))
    with patch.object(caption_runtime.AutoProcessor, 'from_pretrained', return_value=processor), \
         patch.object(caption_runtime, 'load_checkpoint_in_model', side_effect=RuntimeError('bad shard')):
        with pytest.raises(RuntimeError, match='bad shard'):
            with session:
                caption_runtime.ModelHandler(tmp_path, device='cpu')
    assert session.process.poll() is not None
    assert not session.state_dir.exists()
