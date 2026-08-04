"""Patch nn.Linear.forward to use int8_linear_forward when weights are int8_tensorwise."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from h3_infer.int8_linear import int8_linear_forward, is_int8_linear

_PATCHED = "_h3_int8_patched"
_DTYPE = "_h3_int8_compute_dtype"


def patch_linear(module: nn.Linear, compute_dtype: torch.dtype) -> None:
    if hasattr(module, _PATCHED):
        setattr(module, _DTYPE, compute_dtype)
        return
    setattr(module, _PATCHED, True)
    setattr(module, _DTYPE, compute_dtype)

    def forward(x: torch.Tensor) -> torch.Tensor:
        dtype = getattr(module, _DTYPE)
        if is_int8_linear(module):
            return int8_linear_forward(x, module, dtype)
        return F.linear(x, module.weight, module.bias)

    module.forward = forward  # type: ignore[method-assign]


def patch_module_int8_linears(
    root: nn.Module, compute_dtype: torch.dtype = torch.bfloat16
) -> int:
    n = 0
    for mod in root.modules():
        if isinstance(mod, nn.Linear) and is_int8_linear(mod):
            patch_linear(mod, compute_dtype)
            n += 1
    return n
