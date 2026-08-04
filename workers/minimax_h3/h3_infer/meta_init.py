"""Meta-device construction and non-persistent buffer materialization for H3 DiT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _load_config_dict(config_dict_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config_dict_or_path, dict):
        return dict(config_dict_or_path)
    path = Path(config_dict_or_path)
    with path.open() as f:
        return json.load(f)


def empty_minimax_h3_transformer(config_dict_or_path: dict[str, Any] | str | Path):
    """Construct MiniMaxH3Transformer3DModel on meta device (no full weight alloc)."""
    from diffusers import MiniMaxH3Transformer3DModel

    raw = _load_config_dict(config_dict_or_path)
    # Prefer local dict: avoid Hub network.
    try:
        from accelerate import init_empty_weights

        with init_empty_weights():
            model = MiniMaxH3Transformer3DModel.from_config(raw)
    except Exception:
        with torch.device("meta"):
            model = MiniMaxH3Transformer3DModel.from_config(raw)
    model.requires_grad_(False)
    return model


def _compute_rope_inv_freq(
    rope_theta: float, rope_freq_dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Match pin: 1 / theta ** (arange(0, 2*dim, 2) / (2*dim))."""
    dims = torch.arange(0, 2 * rope_freq_dim, 2, device=device, dtype=dtype)
    return 1.0 / (rope_theta ** (dims / (2 * rope_freq_dim)))


def materialize_nonpersistent_buffers(
    model: nn.Module, device: torch.device | str
) -> None:
    """Idempotent: create missing/meta non-persistent buffers (e.g. rope.inv_freq).

    Safe if called more than once; only ``load_comfy_int8_dit`` should call it on
    the production load path.
    """
    device = torch.device(device)

    # Preferred: model hook if present.
    rope = getattr(model, "rope", None)
    if rope is not None and hasattr(rope, "reset_inv_freq"):
        try:
            rope.reset_inv_freq()
            inv = getattr(rope, "inv_freq", None)
            if isinstance(inv, torch.Tensor) and inv.device != device:
                rope.inv_freq = inv.to(device=device)
            return
        except Exception:
            pass

    # Fallback: walk modules named *rope* / with inv_freq buffer.
    cfg = getattr(model, "config", None)
    rope_theta = float(getattr(cfg, "rope_theta", 10000.0) if cfg is not None else 10000.0)
    rope_freq_dim = int(getattr(cfg, "rope_freq_dim", 16) if cfg is not None else 16)

    for name, module in model.named_modules():
        # Detect rotary modules that expose inv_freq.
        has_attr = hasattr(module, "inv_freq")
        if not has_attr and "rope" not in name.lower():
            continue
        inv = getattr(module, "inv_freq", None)
        if inv is not None and isinstance(inv, torch.Tensor):
            if inv.device.type != "meta" and inv.device == device:
                continue  # already good
            if inv.device.type != "meta" and inv.numel() > 0:
                # Move existing non-meta buffer if wrong device.
                module.inv_freq = inv.to(device=device)
                continue
        # Allocate / recompute
        inv_freq = _compute_rope_inv_freq(
            rope_theta, rope_freq_dim, device, torch.float32
        )
        # Prefer register_buffer without persistence when supported.
        if hasattr(module, "register_buffer"):
            try:
                module.register_buffer("inv_freq", inv_freq, persistent=False)
            except TypeError:
                module.register_buffer("inv_freq", inv_freq)
        else:
            module.inv_freq = inv_freq

    # Generic: any remaining meta buffers on the module tree.
    for name, buf in list(model.named_buffers(recurse=True)):
        if not isinstance(buf, torch.Tensor):
            continue
        if buf.device.type != "meta":
            continue
        # Zero-init same shape on target device (last resort).
        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            parent, leaf = model, parts[0]
        else:
            parent = model.get_submodule(parts[0])
            leaf = parts[1]
        filled = torch.zeros(buf.shape, device=device, dtype=buf.dtype)
        if hasattr(parent, "register_buffer"):
            try:
                parent.register_buffer(leaf, filled, persistent=False)
            except TypeError:
                parent.register_buffer(leaf, filled)
        else:
            setattr(parent, leaf, filled)
