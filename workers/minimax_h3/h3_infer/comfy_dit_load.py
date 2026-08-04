"""Load Comfy/original int8 ConvRot DiT into a stock MiniMaxH3Transformer3DModel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file

from h3_infer.comfy_dit_g0 import classify_dit_checkpoint, read_safetensors_header
from h3_infer.int8_linear import apply_int8_side_tensors, partition_int8_state_dict
from h3_infer.int8_linear_patch import patch_module_int8_linears
from h3_infer.meta_init import materialize_nonpersistent_buffers
from h3_infer.minimax_h3_convert import (
    MINIMAX_H3_TRANSFORMER_CONFIG,
    SourceLayout,
    convert_transformer_key_with_sides,
    g1_int8_source_coverage,
    get_transformer_key_plan,
    strip_state_dict_prefixes,
)


def _required_target_keys(config: dict[str, Any]) -> set[str]:
    plan = get_transformer_key_plan(config)
    keys: set[str] = set()
    for targets in plan.values():
        for tk, _ in targets:
            keys.add(tk)
    return keys


def load_comfy_int8_dit(
    module: nn.Module,
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    min_source_coverage: float = 0.8,
    compute_dtype: torch.dtype = torch.bfloat16,
    source_layout: SourceLayout = SourceLayout.COMFY_QKV_CONTIGUOUS,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """G0 → convert → assign=True → G3 → int8 sides → patch → materialize → to(device).

    ``materialize_nonpersistent_buffers`` is owned solely by this function for the
    load path (helper itself is idempotent).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    cfg = dict(config or MINIMAX_H3_TRANSFORMER_CONFIG)
    header = read_safetensors_header(path)
    g0 = classify_dit_checkpoint(
        header, time_embed_dim=int(cfg["time_embed_dim"])
    )
    if not g0.compatible_with_stock_diffusers:
        raise RuntimeError(
            f"G0 FAIL {g0.verdict}: {'; '.join(g0.reasons)}"
        )

    state = load_file(str(path), device="cpu")
    state = strip_state_dict_prefixes(state)

    g1 = g1_int8_source_coverage(state, cfg, source_layout=source_layout)
    if g1["shape_errors"]:
        raise RuntimeError(
            f"G1 FAIL shape_errors={g1['shape_errors'][:5]}"
        )
    if g1["source_coverage_ratio"] < min_source_coverage:
        raise RuntimeError(
            f"G1 FAIL source_coverage_ratio={g1['source_coverage_ratio']:.4f} "
            f"< {min_source_coverage} unmapped={g1['unmapped_int8_weights'][:10]}"
        )

    converted = convert_transformer_key_with_sides(
        state, cfg, source_layout=source_layout
    )

    module.requires_grad_(False)
    weights, side = partition_int8_state_dict(converted)
    incompatible = module.load_state_dict(weights, strict=False, assign=True)

    required = _required_target_keys(cfg)
    loaded = set(weights.keys())
    # Only require keys that are Linear/weight targets present in module.state_dict plan
    # intersection with what convert produced for planned names.
    module_keys = set(module.state_dict().keys())
    missing_required = sorted(
        k for k in required if k in module_keys and k not in loaded
    )
    # Also: planned targets that convert should have produced from present sources.
    # Hard fail if any required module key that appears in converted plan is missing.
    if missing_required:
        raise RuntimeError(
            f"G3 FAIL missing {len(missing_required)} required keys e.g. "
            f"{missing_required[:10]}"
        )

    n_int8 = apply_int8_side_tensors(module, side, weights)
    n_patched = patch_module_int8_linears(module, compute_dtype=compute_dtype)

    # Sole materialize call site for load path.
    materialize_nonpersistent_buffers(module, device)
    module.to(device)

    return {
        "g0_verdict": g0.verdict,
        "g1": g1,
        "int8_layers": n_int8,
        "patched_linears": n_patched,
        "unexpected_keys": list(incompatible.unexpected_keys),
        "missing_keys_soft": list(incompatible.missing_keys),
        "source_layout": source_layout.value,
        "device": str(device),
    }
