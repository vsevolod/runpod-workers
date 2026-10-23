"""Request-scoped MOSS Edit conditioning and local-only public adapter loading.

The reference branch is provided by the pinned GPLv3 upstream implementation
in vendor/moss_edit_model.py; see ../../NOTICE for provenance and license.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from shared.megapixels import resize_references

from .config import EDIT_CONTRACT
from .lora import LoRAError
from .vendor.moss_edit_model import MossEditModel
from .vendor.pipeline_moss_image import MOSSImageTransformer, _validate_skip_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput


class _Editor(MossEditModel):
    """SLG skips only body blocks: the cached t=0 reference branch stays fixed."""
    skip_layers = frozenset()

    def _model_stamp(self):
        parameters, buffers, settings, autocast = super()._model_stamp()
        # Accelerate calls .to(execution_device) before every DiT forward.
        # RoPE reconstructs its derived FP32 frequencies, and .to(cpu:0)
        # can copy immutable adapter buffers. Their object IDs are not weight
        # changes. Keep versions/shape/dtype/device plus strict parameter IDs.
        buffers = tuple((name, stamp[1:]) for name, stamp in buffers
                        if name != 'base_model.rope.bands')
        return parameters, buffers, settings, autocast

    def _body_block(self, layer, body, *args, **kwargs):
        if layer in self.skip_layers:
            return body
        return super()._body_block(layer, body, *args, **kwargs)


@dataclass
class _EditState:
    references: tuple
    editor: _Editor | None = None
    cache: object = None


class MOSSImageEditTransformer(MOSSImageTransformer):
    """Keep the original forward for generation and its Accelerate hooks intact."""
    _edit_state = None

    @contextmanager
    def editing(self, references):
        if self._edit_state is not None or not 1 <= len(references) <= 4:
            raise ValueError('MOSS Edit requires 1..4 references and no active edit request')
        # A plain dataclass avoids registering an editor containing this model
        # as a child module. The whole state, including GPU KV, dies on exit.
        self._edit_state = _EditState(tuple(references))
        try:
            yield
        finally:
            self._edit_state = None

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_attention_mask=None, return_dict=True, skip_layers=None):
        state = self._edit_state
        if state is None:
            return super().forward(hidden_states, timestep, encoder_hidden_states,
                                   encoder_attention_mask, return_dict, skip_layers)
        skipped = _validate_skip_layers(skip_layers, len(self.layers))
        if state.editor is None:
            # Build after the outer forward's Accelerate pre-hook has moved the
            # base weights to the execution device, and after LoRA activation.
            state.editor = _Editor(self).eval()
            state.references = tuple(ref.to(hidden_states.device) for ref in state.references)
            state.cache = state.editor.prepare_cache(state.references)
        state.editor.skip_layers = frozenset(self.layers[index] for index in skipped)
        sample = state.editor(
            hidden_states, torch.as_tensor(timestep, device=hidden_states.device) / 1000.,
            encoder_hidden_states, state.references, encoder_attention_mask, state.cache,
        )
        return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)


@torch.no_grad()
def encode_references(pipe, images, megapixels):
    """FLUX.2 posterior mode, 2x2 channel packing and FP32 BN normalization."""
    references = resize_references(images, megapixels, multiple=16, max_side=4096)
    latents = []
    try:
        for image in references:
            pixels = pipe.image_processor.preprocess(image).to(device=pipe._execution_device, dtype=pipe.vae.dtype)
            z = pipe.vae.encode(pixels).latent_dist.mode()
            batch, channels, height, width = z.shape
            z = z.reshape(batch, channels, height // 2, 2, width // 2, 2)
            z = z.permute(0, 1, 3, 5, 2, 4).reshape(batch, 4 * channels, height // 2, width // 2).float()
            bn, eps = pipe.vae.bn, pipe.vae.config.batch_norm_eps
            z = (z - bn.running_mean.to(z).view(1, -1, 1, 1)) / (
                bn.running_var.to(z) + eps).sqrt().view(1, -1, 1, 1)
            if not torch.isfinite(z).all():
                raise ValueError('MOSS Edit VAE encoding produced nonfinite reference latents')
            latents.append(z.cpu())
    finally:
        # Encoding calls the VAE before the regular text->DiT->VAE chain.
        # Reset/offload it now so it cannot remain on GPU alongside Qwen/DiT.
        pipe.maybe_free_model_hooks()
    return latents


def load_edit_adapter(directory, model):
    """Validate the two-file bundle, then convert A/B into our scoped LoRA format."""
    root = Path(directory)
    try:
        config = json.loads((root / 'adapter_config.json').read_text())
        if config['format'] != 'moss-edit-lora' or config['format_version'] != 1:
            raise LoRAError('Unsupported MOSS Edit adapter format/version')
        metadata = config['metadata']
        if metadata['edit_contract'] != EDIT_CONTRACT:
            raise LoRAError('Unsupported MOSS Edit attention contract')
        if any(model.config.get(key) != value for key, value in metadata['model_config'].items()):
            raise LoRAError('MOSS Edit model configuration mismatch')
        modules = config['modules']
        if not modules or set(config['targets']) != set(modules):
            raise LoRAError('MOSS Edit target configuration mismatch')
        weights = root / 'adapter_model.safetensors'
        with weights.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != config['weights_sha256']:
                raise LoRAError('MOSS Edit weights checksum mismatch')
        state = {}
        with safe_open(str(weights), framework='pt', device='cpu') as bundle:
            expected = {k: v for k, v in config.items() if k != 'weights_sha256'}
            if json.loads((bundle.metadata() or {})['adapter_config']) != expected:
                raise LoRAError('MOSS Edit embedded configuration mismatch')
            if set(bundle.keys()) != {name + suffix for name in modules for suffix in ('.lora_A', '.lora_B')}:
                raise LoRAError('MOSS Edit weight keys mismatch')
            for name, options in modules.items():
                rank, alpha = options['rank'], options['alpha']
                base = model.get_submodule(name)
                if (type(rank) is not int or rank < 1 or not math.isfinite(alpha) or alpha <= 0
                        or options['dropout'] != 0 or not isinstance(base, torch.nn.Linear)
                        or (options['in_features'], options['out_features']) != (base.in_features, base.out_features)):
                    raise LoRAError(f'Invalid MOSS Edit target: {name}')
                down, up = bundle.get_tensor(name + '.lora_A'), bundle.get_tensor(name + '.lora_B')
                if (tuple(down.shape) != (rank, base.in_features) or tuple(up.shape) != (base.out_features, rank)
                        or str(down.dtype).removeprefix('torch.') != options['param_dtype'] or up.dtype != down.dtype
                        or not down.is_floating_point() or not torch.isfinite(down).all() or not torch.isfinite(up).all()):
                    raise LoRAError(f'Invalid MOSS Edit tensors: {name}')
                key = 'diffusion_model.' + name
                state[key + '.lora_down.weight'] = down
                state[key + '.lora_up.weight'] = up
                state[key + '.alpha'] = torch.tensor(float(alpha))
        return state
    except (KeyError, TypeError, AttributeError, ValueError) as err:
        if isinstance(err, LoRAError):
            raise
        raise LoRAError(f'Invalid MOSS Edit adapter: {err}') from err
