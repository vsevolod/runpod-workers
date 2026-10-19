"""Strict, unmerged Kohya adapters scoped to a single inference request."""

from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRAError(ValueError):
    """Invalid adapter contents, safe to report to a client."""


class LoRALinear(nn.Module):
    def __init__(self, base, down, up, scale):
        super().__init__()
        self.base = base
        self.register_buffer('down', down.to(device=base.weight.device, dtype=base.weight.dtype), persistent=False)
        self.register_buffer('up', up.to(device=base.weight.device, dtype=base.weight.dtype), persistent=False)
        self.scale = scale

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.down), self.up) * self.scale


@contextmanager
def activate_lora(model, state, *, strength):
    prefix = 'diffusion_model.'
    suffix = '.lora_down.weight'
    names = [key[len(prefix):-len(suffix)] for key in state if key.startswith(prefix) and key.endswith(suffix)]
    prepared, consumed = [], set()
    if not names or not math.isfinite(strength):
        raise LoRAError('MOSS LoRA has no matching linear layers or invalid strength')
    for name in names:
        key = prefix + name
        keys = (key + suffix, key + '.lora_up.weight', key + '.alpha')
        try:
            down, up, alpha = (state[k] for k in keys)
            parent_name, _, child = name.rpartition('.')
            parent = model.get_submodule(parent_name)
            base = getattr(parent, child)
            valid = (isinstance(base, nn.Linear) and down.ndim == up.ndim == 2 and down.shape[0] > 0
                     and tuple(down.shape) == (up.shape[1], base.in_features)
                     and up.shape[0] == base.out_features and alpha.numel() == 1
                     and down.is_floating_point() and up.is_floating_point()
                     and math.isfinite(float(alpha)))
        except (KeyError, AttributeError, ValueError, RuntimeError) as err:
            raise LoRAError(f'Invalid MOSS LoRA target: {name}') from err
        if not valid:
            raise LoRAError(f'Invalid MOSS LoRA shapes/alpha: {name}')
        consumed.update(keys)
        prepared.append((parent, child, base, down, up, strength * float(alpha) / down.shape[0]))
    if set(state) != consumed:
        raise LoRAError('MOSS LoRA contains unsupported or incomplete tensors')
    installed = []
    try:
        for parent, child, base, down, up, scale in prepared:
            setattr(parent, child, LoRALinear(base, down, up, scale))
            installed.append((parent, child, base))
        yield
    finally:
        for parent, child, base in reversed(installed):
            setattr(parent, child, base)
