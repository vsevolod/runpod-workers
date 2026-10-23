"""Real CPU edit attention, cache ownership and checkpoint-format validation."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

sys.path[:0] = [str(Path(__file__).parents[2]), str(Path(__file__).parents[1])]


def save_edit_adapter(directory, model, *, corrupt=False):
    """Small public-format bundle, including embedded metadata and checksum."""
    from safetensors.torch import save_file
    from moss_image_infer.config import EDIT_CONTRACT
    name = 'layers.0.proj.q'
    layer = model.get_submodule(name)
    config = {
        'format': 'moss-edit-lora', 'format_version': 1,
        'metadata': {'edit_contract': EDIT_CONTRACT,
                     'model_config': {k: v for k, v in model.config.items() if not k.startswith('_')}},
        'targets': [name],
        'modules': {name: {'alpha': 2., 'rank': 2, 'dropout': 0.,
                           'in_features': layer.in_features, 'out_features': layer.out_features,
                           'param_dtype': 'float32'}},
    }
    state = {name + '.lora_A': torch.ones(2, layer.in_features),
             name + '.lora_B': torch.full((layer.out_features, 2), .05)}
    weights = directory / 'adapter_model.safetensors'
    save_file(state, str(weights), metadata={'adapter_config': json.dumps(config)})
    config['weights_sha256'] = hashlib.sha256(weights.read_bytes()).hexdigest()
    if corrupt:
        config['metadata']['edit_contract'] = 'wrong-contract'
    (directory / 'adapter_config.json').write_text(json.dumps(config))


class EditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        from moss_image_infer.edit import MOSSImageEditTransformer
        torch.manual_seed(4)
        return MOSSImageEditTransformer(dim=16, ffn_dim=32, num_heads=2, num_layers=2,
                                        text_dim=8, freq_dim=8).eval()

    def test_reference_encoding_preserves_flux2_channel_packing_and_bn_math(self):
        from moss_image_infer.edit import encode_references
        from diffusers.image_processor import VaeImageProcessor
        from PIL import Image
        events = []
        def encode(pixels):
            self.assertEqual(tuple(pixels.shape), (1, 3, 32, 32))
            self.assertEqual(pixels.dtype, torch.float32)
            self.assertEqual(pixels[0, :, 0, 0].tolist(), [1., -1., -1.])
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: torch.arange(16.).reshape(1, 1, 4, 4)))
        pipe = SimpleNamespace(
            image_processor=VaeImageProcessor(vae_scale_factor=16), _execution_device=torch.device('cpu'),
            vae=SimpleNamespace(dtype=torch.float32, encode=encode, config=SimpleNamespace(batch_norm_eps=0.),
                                bn=SimpleNamespace(running_mean=torch.tensor([0., 1., 2., 3.]),
                                                   running_var=torch.tensor([1., 4., 9., 16.]))),
            maybe_free_model_hooks=lambda: events.append('offloaded'),
        )
        actual = encode_references(pipe, [Image.new('RGB', (32, 32), 'red')], .262144)[0]
        expected = torch.tensor([[[[0., 2.], [8., 10.]], [[0., 1.], [4., 5.]],
                                   [[2/3, 4/3], [10/3, 4.]], [[.5, 1.], [2.5, 3.]]]])
        torch.testing.assert_close(actual, expected)
        self.assertEqual(events, ['offloaded'])

    def test_edit_conditions_on_references_and_cache_never_leaks_to_generation(self):
        model = self.model()
        image, text = torch.randn(1, 128, 2, 3), torch.randn(1, 2, 8)
        reference = torch.randn(1, 128, 3, 2)
        with torch.no_grad():
            generation = model(image, 700., text).sample
            with model.editing([reference]):
                first = model(image, 700., text).sample
                cache = model._edit_state.cache
                model(image, 200., text, skip_layers=[0])
                self.assertIs(model._edit_state.cache, cache)
                skipped = model(image, 700., text, skip_layers=[0, 1]).sample
                video = model.video_embedder(image.flatten(2).transpose(1, 2))
                _, head = model._edit_state.editor._time(.7, image.device)
                expected = model.final_norm_video(video) * (1 + head[:, 1, None]) + head[:, 0, None]
                expected = model.final_linear_video(expected).transpose(1, 2).reshape_as(image)
                torch.testing.assert_close(skipped, expected)
            self.assertIsNone(model._edit_state)
            with model.editing([reference + 2.]):
                second = model(image, 700., text).sample
            self.assertFalse(torch.allclose(first, second))
            torch.testing.assert_close(model(image, 700., text).sample, generation, rtol=0, atol=0)
            try:
                with model.editing([reference]):
                    model(image, 700., text)
                    raise RuntimeError('decode failed')
            except RuntimeError:
                pass
            self.assertIsNone(model._edit_state)

    def test_cached_edit_matches_dense_masked_attention_for_multiple_references(self):
        from moss_image_infer.vendor.pipeline_moss_image import _rotate, _situ
        from torch.nn import functional as F
        model = self.model()
        references = [torch.randn(1, 128, 2, 2), torch.randn(1, 128, 1, 3)]
        image, text = torch.randn(1, 128, 2, 3), torch.randn(1, 3, 8)
        # Independent full [text, ref, target] forward, with no KV cache.
        # Ref queries see refs only, while body queries see every key.
        def dense():
            editor = model._edit_state.editor
            prefix, ref_rope = editor._reference_inputs(tuple(references))
            rope = model.rope(2, 3, 3, image.device)
            video = model.video_embedder(image.flatten(2).transpose(1, 2))
            context = model.text_embedder(text, rope[:, 6:])
            x = torch.cat((context, prefix, video), 1)
            rope = torch.cat((rope[:, 6:], ref_rope, rope[:, :6]), 1)
            ref_mod, _ = editor._time(0., image.device)
            body_mod, head = editor._time(.7, image.device)
            nt, nr, nv = 3, 7, 6
            mod = torch.cat((body_mod.unsqueeze(2).expand(-1, -1, nt, -1),
                             ref_mod.unsqueeze(2).expand(-1, -1, nr, -1),
                             body_mod.unsqueeze(2).expand(-1, -1, nv, -1)), 2)
            visibility = torch.ones(nt + nr + nv, nt + nr + nv, dtype=torch.bool)
            visibility[nt:nt + nr] = False
            visibility[nt:nt + nr, nt:nt + nr] = True
            for layer in model.layers:
                y = layer.norm1(x) * (1 + mod[:, 1]) + mod[:, 0]
                q = layer.q_norm(layer.proj['q'](y).unflatten(-1, (2, 8)))
                k = layer.k_norm(layer.proj['k'](y).unflatten(-1, (2, 8)))
                v = layer.proj['v'](y).unflatten(-1, (2, 8)).transpose(1, 2)
                attention = F.scaled_dot_product_attention(
                    _rotate(q, rope).transpose(1, 2), _rotate(k, rope).transpose(1, 2), v,
                    attn_mask=visibility).transpose(1, 2).flatten(2)
                x = x + layer.proj['o'](attention) * mod[:, 2]
                y = layer.norm2(x) * (1 + mod[:, 4]) + mod[:, 3]
                h, g = layer.proj['ffn_up_gate'](y).chunk(2, -1)
                x = x + layer.proj['ffn_down'](_situ(h, g)) * mod[:, 5]
            x = model.final_norm_video(x[:, -6:])
            x = x * (1 + head[:, 1, None]) + head[:, 0, None]
            return model.final_linear_video(x).transpose(1, 2).reshape_as(image)
        with torch.no_grad(), model.editing(references):
            actual = model(image, 700., text).sample
            torch.testing.assert_close(actual, dense(), rtol=2e-5, atol=2e-6)
            # Padded text cannot influence edit conditioning.
            padded = torch.cat((text, torch.full((1, 1, 8), 1000.)), 1)
            torch.testing.assert_close(model(image, 700., padded, torch.tensor([[1, 1, 1, 0]])).sample, actual)

    def test_edit_adapter_loads_public_format_without_mutating_base_weights(self):
        from moss_image_infer.edit import load_edit_adapter
        from moss_image_infer.lora import activate_lora, LoRAError
        model = self.model()
        original = model.layers[0].proj['q']
        x = torch.ones(1, 16)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_edit_adapter(directory, model)
            state = load_edit_adapter(directory, model)
            expected = original(x) + 1.6
            with activate_lora(model, state, strength=1.):
                torch.testing.assert_close(model.layers[0].proj['q'](x), expected)
            self.assertIs(model.layers[0].proj['q'], original)
            save_edit_adapter(directory, model, corrupt=True)
            with self.assertRaises(LoRAError):
                load_edit_adapter(directory, model)
            save_edit_adapter(directory, model)
            with (directory / 'adapter_model.safetensors').open('ab') as handle:
                handle.write(b'corrupt')
            with self.assertRaisesRegex(LoRAError, 'checksum'):
                load_edit_adapter(directory, model)
