"""INT8 tensorwise (W8A8) linear helpers for int8_tensorwise / ConvRot DiT weights.

Runtime path:
  * Prefer CUDA ``torch._int_mm`` with dynamic per-token activation quant (true INT8 GEMM).
  * Fall back to dequant → BF16/FP32 ``F.linear`` on CPU / small batches / missing kernels.

This is intentionally independent of ComfyUI; storage matches stock ``int8_tensorwise``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .convrot import build_hadamard, rotate_activation

logger = logging.getLogger(__name__)

DEFAULT_CONVROT_GROUPSIZE = 256
# Below this token count, dequant matmul is often cheaper / more accurate enough.
_INT_MM_MIN_TOKENS = 16

_SIDE_SUFFIXES = (".weight_scale", ".comfy_quant", ".input_scale")


def is_int8_linear(module: nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    scale = getattr(module, "weight_scale", None)
    return (
        isinstance(module, nn.Linear)
        and isinstance(weight, torch.Tensor)
        and weight.dtype == torch.int8
        and isinstance(scale, torch.Tensor)
    )


def quantize_int8(x: Tensor, scale: float | Tensor) -> Tensor:
    return x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)


def quantize_int8_axiswise(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale


def parse_comfy_quant_marker(marker: Tensor | None) -> dict[str, Any]:
    if marker is None:
        return {}
    try:
        if marker.dtype == torch.uint8 or marker.dtype == torch.int8:
            raw = bytes(marker.detach().cpu().tolist())
        else:
            raw = bytes(int(v) & 0xFF for v in marker.detach().cpu().flatten().tolist())
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse comfy_quant marker: %s", exc)
        return {}


def partition_int8_state_dict(
    state: dict[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Split weight/bias tensors from quant side-car keys."""
    weights: dict[str, Tensor] = {}
    side: dict[str, Tensor] = {}
    for key, value in state.items():
        if any(key.endswith(suffix) for suffix in _SIDE_SUFFIXES):
            side[key] = value
        else:
            weights[key] = value
    return weights, side


def state_dict_has_int8_tensorwise(state: dict[str, Tensor]) -> bool:
    for key, value in state.items():
        if not key.endswith(".weight"):
            continue
        if value.dtype != torch.int8:
            continue
        scale_key = key[: -len(".weight")] + ".weight_scale"
        if scale_key in state:
            return True
    return False


def materialize_int8_linear(
    module: nn.Linear,
    *,
    weight: Tensor,
    weight_scale: Tensor,
    bias: Tensor | None = None,
    comfy_quant: Tensor | None = None,
) -> None:
    """Attach int8 weight + scale (+ optional ConvRot flags) onto an nn.Linear."""
    if weight.dtype != torch.int8:
        raise TypeError(f"Expected int8 weight, got {weight.dtype}")
    if weight_scale is None:
        raise ValueError("weight_scale is required for int8_tensorwise layers")

    module.weight = nn.Parameter(weight, requires_grad=False)
    if bias is not None:
        module.bias = nn.Parameter(bias, requires_grad=False)
    else:
        module.bias = None

    scale = weight_scale.detach().float()
    if scale.numel() == 1:
        scale = scale.reshape(1, 1)
    elif scale.dim() == 1:
        scale = scale.reshape(-1, 1)
    module.register_buffer("weight_scale", scale)

    conf = parse_comfy_quant_marker(comfy_quant)
    use_convrot = bool(conf.get("convrot", False))
    group_size = int(conf.get("convrot_groupsize", DEFAULT_CONVROT_GROUPSIZE))
    if use_convrot and module.in_features % group_size != 0:
        # Checkpoint should not mark these; be safe and skip online rotation.
        logger.warning(
            "ConvRot marker present but in_features=%d not divisible by %d; disabling",
            module.in_features,
            group_size,
        )
        use_convrot = False
    module._use_convrot = use_convrot  # type: ignore[attr-defined]
    module._convrot_groupsize = group_size  # type: ignore[attr-defined]
    module._is_int8_tensorwise = True  # type: ignore[attr-defined]


def apply_int8_side_tensors(
    model: nn.Module,
    side: dict[str, Tensor],
    weight_state: dict[str, Tensor],
) -> int:
    """Materialize int8 metadata for Linear modules present in weight_state.

    Call after int8 ``weight`` tensors have been assigned onto the module.
    Returns number of int8 layers configured.
    """
    modules = dict(model.named_modules())
    configured = 0
    for key, tensor in weight_state.items():
        if not key.endswith(".weight") or tensor.dtype != torch.int8:
            continue
        prefix = key[: -len(".weight")]
        module = modules.get(prefix)
        if not isinstance(module, nn.Linear):
            continue
        scale = side.get(f"{prefix}.weight_scale")
        if scale is None:
            raise RuntimeError(
                f"INT8 weight at {key!r} is missing matching weight_scale"
            )
        marker = side.get(f"{prefix}.comfy_quant")
        bias = weight_state.get(f"{prefix}.bias")
        # Weight already assigned via load_state_dict; re-bind scale/flags.
        materialize_int8_linear(
            module,
            weight=module.weight.detach(),
            weight_scale=scale,
            bias=module.bias.detach() if module.bias is not None else bias,
            comfy_quant=marker,
        )
        configured += 1
    return configured


def _can_use_int_mm(x_2d: Tensor, weight: Tensor) -> bool:
    if not x_2d.is_cuda or not weight.is_cuda:
        return False
    if not hasattr(torch, "_int_mm"):
        return False
    if x_2d.shape[0] < _INT_MM_MIN_TOKENS:
        return False
    # torch._int_mm requires contiguous int8 inputs with compatible shapes.
    return x_2d.shape[-1] == weight.shape[-1]


@torch.no_grad()
def int8_linear_forward(
    x: Tensor,
    module: nn.Linear,
    compute_dtype: torch.dtype,
) -> Tensor:
    """W8A8 linear with optional online ConvRot on activations."""
    weight = module.weight
    if weight.dtype != torch.int8:
        raise TypeError("int8_linear_forward requires int8 weights")

    w_scale = getattr(module, "weight_scale", None)
    if w_scale is None:
        raise RuntimeError("int8 linear is missing weight_scale buffer")

    bias = module.bias
    x_shape = x.shape
    x_2d = x.reshape(-1, x_shape[-1])
    if x_2d.dtype != compute_dtype:
        x_2d = x_2d.to(compute_dtype)

    if getattr(module, "_use_convrot", False):
        group_size = int(
            getattr(module, "_convrot_groupsize", DEFAULT_CONVROT_GROUPSIZE)
        )
        h = build_hadamard(group_size, device=x_2d.device, dtype=x_2d.dtype)
        x_2d = rotate_activation(x_2d, h, group_size)

    if isinstance(w_scale, torch.Tensor) and w_scale.device != x_2d.device:
        w_scale = w_scale.to(device=x_2d.device, non_blocking=True)
    w_scale = w_scale.float()

    if _can_use_int_mm(x_2d, weight):
        x_8, x_scale = quantize_int8_axiswise(x_2d, dim=-1)
        # torch._int_mm(a, b): a [m,k] int8, b [k,n] int8 → [m,n] int32
        w_t = weight.t().contiguous()
        res = torch._int_mm(x_8.contiguous(), w_t)
        # per-token x_scale [m,1] and per-out-row w_scale [out,1] → [1,out]
        y = res.float().mul_(x_scale).mul_(w_scale.reshape(1, -1))
        y = y.to(compute_dtype)
        if bias is not None:
            y = y + bias.to(device=y.device, dtype=compute_dtype)
    else:
        w_float = weight.to(dtype=torch.float32) * w_scale.to(dtype=torch.float32)
        w_float = w_float.to(dtype=compute_dtype)
        bias_t = None if bias is None else bias.to(device=x_2d.device, dtype=compute_dtype)
        y = F.linear(x_2d, w_float, bias_t)

    return y.reshape(*x_shape[:-1], y.shape[-1])
