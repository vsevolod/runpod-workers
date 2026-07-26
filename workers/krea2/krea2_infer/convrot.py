"""Group-wise regular Hadamard rotation for INT8 ConvRot.

Based on ConvRot / QuaRot style block-diagonal rotation used by ComfyUI
int8_tensorwise checkpoints (group size typically 256).
"""

from __future__ import annotations

import math

import torch

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a normalized regular orthogonal Hadamard matrix (power of 4)."""
    cache_key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    # Base H4 (symmetric); Kronecker construction preserves symmetry + orthogonality.
    h4 = torch.tensor(
        [
            [1, 1, 1, -1],
            [1, 1, -1, 1],
            [1, -1, 1, 1],
            [-1, 1, 1, 1],
        ],
        dtype=dtype,
        device=device,
    )

    h = h4
    current = 4
    while current < size:
        h = torch.kron(h, h4)
        current *= 4

    h_normalized = h / (size**0.5)
    _HADAMARD_CACHE[cache_key] = h_normalized
    return h_normalized


def rotate_weight(
    weight: torch.Tensor,
    h: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Offline weight rotation: W_rot = W @ H^T (row groups)."""
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by group_size {group_size}")
    n_groups = in_f // group_size
    w_grouped = weight.view(out_f, n_groups, group_size)
    h_t = h.T.to(dtype=weight.dtype, device=weight.device)
    return torch.matmul(w_grouped, h_t).reshape(out_f, in_f)


def rotate_activation(
    x: torch.Tensor,
    h: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Online activation rotation: x_rot = x @ H (last-dim groups)."""
    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    n_groups = features // group_size
    x_grouped = x.view(*orig_shape[:-1], n_groups, group_size)
    h_dev = h.to(dtype=x.dtype, device=x.device)
    return torch.matmul(x_grouped, h_dev).view(orig_shape)
