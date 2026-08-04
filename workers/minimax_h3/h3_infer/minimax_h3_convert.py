"""MiniMax-H3 transformer key conversion for Level-1 hybrid / official shards.

Vendored and adapted from diffusers pin
``abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc``
``scripts/convert_minimax_h3_to_diffusers.py`` (transformer path only).

Adds:
* explicit ``SourceLayout`` for QKV pre-step (reorder vs identity)
* int8 side tensors (``weight_scale``, ``comfy_quant``) following the same row ops
* G1 coverage over *all* int8 ``.weight`` tensors (unknown keys lower the ratio)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Configs (from pinned converter)
# ---------------------------------------------------------------------------

MINIMAX_H3_TRANSFORMER_CONFIG: dict[str, Any] = {
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "hidden_size": 5376,
    "num_layers": 50,
    "num_refiner_layers": 2,
    "ffn_dim": 14336,
    "in_channels": 24,
    "audio_in_channels": 32,
    "patch_size": [1, 2, 2],
    "text_dim": 5120,
    "freq_dim": 256,
    "time_embed_hidden_dim": 5376,
    "time_embed_dim": 2688,
    "rope_freq_dim": 16,
    "rope_theta": 10000.0,
    "norm_eps": 1e-05,
    "qk_norm_eps": 1e-05,
    "final_norm_eps": 1e-05,
}

MINIMAX_H3_TEST_TRANSFORMER_CONFIG: dict[str, Any] = {
    **MINIMAX_H3_TRANSFORMER_CONFIG,
    "num_attention_heads": 2,
    "attention_head_dim": 32,
    "hidden_size": 64,
    "num_layers": 2,
    "num_refiner_layers": 2,
    "ffn_dim": 128,
    "text_dim": 48,
    "freq_dim": 16,
    "time_embed_hidden_dim": 64,
    "time_embed_dim": 32,
    "rope_freq_dim": 4,
}

MINIMAX_H3_TRANSFORMER_DROPPED_KEYS = ("rope.inv_freq",)

# Common Comfy single-file prefixes (strip before convert).
_COMFY_PREFIXES = (
    "diffusion_model.",
    "model.diffusion_model.",
)


class SourceLayout(str, Enum):
    """How fused ``*.attn.qkv_proj.weight`` rows are stored on disk."""

    # Official MiniMax shards: per-head interleaved [h0:qkv, h1:qkv, ...].
    OFFICIAL_RAW_INTERLEAVED = "official_raw_interleaved"

    # Comfy-Org single-file DiT: already contiguous [q_all; k_all; v_all].
    COMFY_QKV_CONTIGUOUS = "comfy_qkv_contiguous"


# ---------------------------------------------------------------------------
# Core transforms (pinned converter)
# ---------------------------------------------------------------------------


def reorder_interleaved_qkv(
    weight: torch.Tensor, num_attention_heads: int, attention_head_dim: int
) -> torch.Tensor:
    """Reorder raw per-head-interleaved fused QKV into ``[q_all; k_all; v_all]``."""
    expected_rows = num_attention_heads * 3 * attention_head_dim
    if weight.shape[0] != expected_rows:
        raise ValueError(
            f"fused qkv weight has {weight.shape[0]} rows, expected "
            f"{expected_rows} = {num_attention_heads} heads * 3 * {attention_head_dim}."
        )
    grouped = weight.reshape(
        num_attention_heads, 3 * attention_head_dim, *weight.shape[1:]
    )
    query, key, value = grouped.split(attention_head_dim, dim=1)
    return torch.cat(
        [
            tensor.reshape(num_attention_heads * attention_head_dim, *weight.shape[1:])
            for tensor in (query, key, value)
        ],
        dim=0,
    )


def split_fused_qkv(
    weight: torch.Tensor, num_attention_heads: int, attention_head_dim: int
) -> tuple[torch.Tensor, ...]:
    """Split contiguous ``[q_all; k_all; v_all]`` into to_q / to_k / to_v."""
    inner_dim = num_attention_heads * attention_head_dim
    if weight.shape[0] != 3 * inner_dim:
        raise ValueError(
            f"fused qkv weight has {weight.shape[0]} rows, expected "
            f"{3 * inner_dim} = 3 * {num_attention_heads} heads * {attention_head_dim}."
        )
    query, key, value = weight.split(inner_dim, dim=0)
    return tuple(tensor.contiguous() for tensor in (query, key, value))


def get_transformer_key_plan(
    config: dict[str, Any],
) -> dict[str, list[tuple[str, list[int]]]]:
    """Map every original transformer key to diffusers key(s) + shapes."""
    hidden_size = config["hidden_size"]
    heads = config["num_attention_heads"]
    head_dim = config["attention_head_dim"]
    inner_dim = heads * head_dim
    ffn_dim = config["ffn_dim"]
    time_embed_dim = config["time_embed_dim"]
    video_patch_dim = (
        config["in_channels"]
        * config["patch_size"][0]
        * config["patch_size"][1]
        * config["patch_size"][2]
    )

    plan: dict[str, list[tuple[str, list[int]]]] = {
        "video_patch_proj.weight": [("proj_in.weight", [hidden_size, video_patch_dim])],
        "video_patch_proj.bias": [("proj_in.bias", [hidden_size])],
        "audio_patch_proj.weight": [
            ("audio_proj_in.weight", [hidden_size, config["audio_in_channels"]])
        ],
        "audio_patch_proj.bias": [("audio_proj_in.bias", [hidden_size])],
        "condition_proj.weight": [
            ("context_embedder.weight", [hidden_size, config["text_dim"]])
        ],
        "condition_proj.bias": [("context_embedder.bias", [hidden_size])],
        "time_embedder.proj_in.weight": [
            (
                "time_embedder.linear_1.weight",
                [config["time_embed_hidden_dim"], config["freq_dim"]],
            )
        ],
        "time_embedder.proj_in.bias": [
            ("time_embedder.linear_1.bias", [config["time_embed_hidden_dim"]])
        ],
        "time_embedder.proj_out.weight": [
            (
                "time_embedder.linear_2.weight",
                [time_embed_dim, config["time_embed_hidden_dim"]],
            )
        ],
        "time_embedder.proj_out.bias": [("time_embedder.linear_2.bias", [time_embed_dim])],
        "token_refiner.final_norm.weight": [
            ("token_refiner.final_norm.weight", [hidden_size])
        ],
        "final_layer.norm.weight": [("norm_out.norm.weight", [hidden_size])],
        "final_layer.adaln_proj.linear.weight": [
            ("norm_out.linear.weight", [2 * hidden_size, time_embed_dim])
        ],
        "final_layer.adaln_proj.linear.bias": [
            ("norm_out.linear.bias", [2 * hidden_size])
        ],
        "final_layer.video_out.weight": [
            ("proj_out.weight", [video_patch_dim, hidden_size])
        ],
        "final_layer.video_out.bias": [("proj_out.bias", [video_patch_dim])],
        "final_layer.audio_out.weight": [
            ("audio_proj_out.weight", [config["audio_in_channels"], hidden_size])
        ],
        "final_layer.audio_out.bias": [
            ("audio_proj_out.bias", [config["audio_in_channels"]])
        ],
    }
    for key in MINIMAX_H3_TRANSFORMER_DROPPED_KEYS:
        plan[key] = []

    block_specs = [
        ("blocks", "transformer_blocks", config["num_layers"], True),
        (
            "token_refiner.blocks",
            "token_refiner.refiner_blocks",
            config["num_refiner_layers"],
            False,
        ),
    ]
    for source_prefix, target_prefix, num_layers, has_adaln in block_specs:
        for i in range(num_layers):
            source = f"{source_prefix}.{i}"
            target = f"{target_prefix}.{i}"
            plan[f"{source}.norm1.weight"] = [(f"{target}.norm1.weight", [hidden_size])]
            plan[f"{source}.norm2.weight"] = [(f"{target}.norm2.weight", [hidden_size])]
            plan[f"{source}.attn.qkv_proj.weight"] = [
                (f"{target}.attn.to_q.weight", [inner_dim, hidden_size]),
                (f"{target}.attn.to_k.weight", [inner_dim, hidden_size]),
                (f"{target}.attn.to_v.weight", [inner_dim, hidden_size]),
            ]
            plan[f"{source}.attn.q_norm.weight"] = [
                (f"{target}.attn.norm_q.weight", [head_dim])
            ]
            plan[f"{source}.attn.k_norm.weight"] = [
                (f"{target}.attn.norm_k.weight", [head_dim])
            ]
            plan[f"{source}.attn.out_proj.weight"] = [
                (f"{target}.attn.to_out.0.weight", [hidden_size, inner_dim])
            ]
            plan[f"{source}.mlp.fc1.weight"] = [
                (f"{target}.ff.net.0.proj.weight", [2 * ffn_dim, hidden_size])
            ]
            plan[f"{source}.mlp.fc2.weight"] = [
                (f"{target}.ff.net.2.weight", [hidden_size, ffn_dim])
            ]
            if has_adaln:
                plan[f"{source}.adaln_proj.linear.weight"] = [
                    (
                        f"{target}.adaln_proj.linear.weight",
                        [6 * 3 * hidden_size, time_embed_dim],
                    )
                ]
                plan[f"{source}.adaln_proj.linear.bias"] = [
                    (f"{target}.adaln_proj.linear.bias", [6 * 3 * hidden_size])
                ]

    return plan


def convert_transformer_key(
    source_key: str, tensor: torch.Tensor, config: dict[str, Any]
) -> list[tuple[str, torch.Tensor]]:
    """Convert one original key/tensor into diffusers key/tensor pair(s).

    Expects QKV already in reference layout ``[q_all; k_all; v_all]``.
    """
    if source_key in MINIMAX_H3_TRANSFORMER_DROPPED_KEYS:
        return []

    target_key = source_key
    if target_key.startswith("token_refiner.blocks."):
        target_key = target_key.replace(
            "token_refiner.blocks.", "token_refiner.refiner_blocks.", 1
        )
    elif target_key.startswith("blocks."):
        target_key = target_key.replace("blocks.", "transformer_blocks.", 1)
    target_key = target_key.replace(
        "time_embedder.proj_in.", "time_embedder.linear_1."
    )
    target_key = target_key.replace(
        "time_embedder.proj_out.", "time_embedder.linear_2."
    )
    target_key = target_key.replace("video_patch_proj.", "proj_in.")
    target_key = target_key.replace("audio_patch_proj.", "audio_proj_in.")
    target_key = target_key.replace("condition_proj.", "context_embedder.")
    target_key = target_key.replace("final_layer.norm.", "norm_out.norm.")
    target_key = target_key.replace(
        "final_layer.adaln_proj.linear.", "norm_out.linear."
    )
    target_key = target_key.replace("final_layer.video_out.", "proj_out.")
    target_key = target_key.replace("final_layer.audio_out.", "audio_proj_out.")
    target_key = target_key.replace(".attn.q_norm.", ".attn.norm_q.")
    target_key = target_key.replace(".attn.k_norm.", ".attn.norm_k.")
    target_key = target_key.replace(".attn.out_proj.", ".attn.to_out.0.")

    if target_key.endswith(".attn.qkv_proj.weight"):
        query, key, value = split_fused_qkv(
            tensor, config["num_attention_heads"], config["attention_head_dim"]
        )
        prefix = target_key.removesuffix("qkv_proj.weight")
        return [
            (f"{prefix}to_q.weight", query),
            (f"{prefix}to_k.weight", key),
            (f"{prefix}to_v.weight", value),
        ]

    if target_key.endswith(".mlp.fc1.weight"):
        gate, value = tensor.chunk(2, dim=0)
        target_key = target_key.replace(".mlp.fc1.weight", ".ff.net.0.proj.weight")
        return [(target_key, torch.cat([value, gate], dim=0).contiguous())]

    target_key = target_key.replace(".mlp.fc2.", ".ff.net.2.")
    return [(target_key, tensor)]


# ---------------------------------------------------------------------------
# Layout-aware conversion + int8 sides
# ---------------------------------------------------------------------------


def strip_comfy_prefix(key: str) -> str:
    for prefix in _COMFY_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def strip_state_dict_prefixes(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {strip_comfy_prefix(k): v for k, v in state.items()}


def _rename_side_key(source_key: str, side_suffix: str) -> str | None:
    """Map original ``*.qkv_proj.weight_scale`` style key to a base weight key."""
    if not source_key.endswith(side_suffix):
        return None
    return source_key[: -len(side_suffix)] + ".weight"


def _apply_qkv_prestep(
    weight: torch.Tensor,
    config: dict[str, Any],
    source_layout: SourceLayout,
) -> torch.Tensor:
    heads = config["num_attention_heads"]
    head_dim = config["attention_head_dim"]
    if source_layout is SourceLayout.OFFICIAL_RAW_INTERLEAVED:
        return reorder_interleaved_qkv(weight, heads, head_dim)
    if source_layout is SourceLayout.COMFY_QKV_CONTIGUOUS:
        return weight
    raise ValueError(f"unknown source_layout: {source_layout!r}")


def _half_swap_rows(tensor: torch.Tensor) -> torch.Tensor:
    gate, value = tensor.chunk(2, dim=0)
    return torch.cat([value, gate], dim=0).contiguous()


def convert_transformer_key_with_sides(
    state: dict[str, torch.Tensor],
    config: dict[str, Any],
    *,
    source_layout: SourceLayout,
) -> dict[str, torch.Tensor]:
    """Convert weights + side tensors. QKV pre-step depends on ``source_layout``."""
    state = strip_state_dict_prefixes(state)
    out: dict[str, torch.Tensor] = {}
    processed_sides: set[str] = set()
    plan = get_transformer_key_plan(config)

    # Process all .weight keys first (primary path).
    weight_keys = [k for k in state if k.endswith(".weight")]
    for source_key in weight_keys:
        tensor = state[source_key]
        if source_key in MINIMAX_H3_TRANSFORMER_DROPPED_KEYS:
            continue

        # QKV fused: layout pre-step then convert_transformer_key split.
        if source_key.endswith(".attn.qkv_proj.weight"):
            w = _apply_qkv_prestep(tensor, config, source_layout)
            pairs = convert_transformer_key(source_key, w, config)
            for tk, tv in pairs:
                out[tk] = tv

            base = source_key[: -len(".weight")]
            scale_key = f"{base}.weight_scale"
            marker_key = f"{base}.comfy_quant"
            if scale_key in state:
                scale = state[scale_key]
                processed_sides.add(scale_key)
                heads = config["num_attention_heads"]
                head_dim = config["attention_head_dim"]
                if scale.dim() >= 1 and scale.shape[0] == tensor.shape[0]:
                    # Same row pre-step + split as the fused weight.
                    if source_layout is SourceLayout.OFFICIAL_RAW_INTERLEAVED:
                        if scale.dim() == 1:
                            # reorder expects [rows, ...]; treat scale as [rows, 1]
                            sc2 = reorder_interleaved_qkv(
                                scale.unsqueeze(-1), heads, head_dim
                            ).squeeze(-1)
                        else:
                            sc2 = reorder_interleaved_qkv(scale, heads, head_dim)
                    else:
                        sc2 = scale
                    inner = heads * head_dim
                    thirds = tuple(sc2.split(inner, dim=0))
                    for (tk, _), sc in zip(pairs, thirds):
                        out[tk[: -len(".weight")] + ".weight_scale"] = sc.contiguous()
                else:
                    # Broadcast / scalar scale: clone onto each split target.
                    for tk, _ in pairs:
                        out[tk[: -len(".weight")] + ".weight_scale"] = scale.clone()

            if marker_key in state:
                processed_sides.add(marker_key)
                marker = state[marker_key]
                for tk, _ in pairs:
                    out[tk[: -len(".weight")] + ".comfy_quant"] = marker.clone()
            continue

        # MLP fc1: half-swap weight (+ scale).
        if source_key.endswith(".mlp.fc1.weight"):
            pairs = convert_transformer_key(source_key, tensor, config)
            for tk, tv in pairs:
                out[tk] = tv
            base = source_key[: -len(".weight")]
            scale_key = f"{base}.weight_scale"
            marker_key = f"{base}.comfy_quant"
            if scale_key in state:
                scale = state[scale_key]
                processed_sides.add(scale_key)
                if scale.dim() >= 1 and scale.shape[0] == tensor.shape[0]:
                    scale_out = _half_swap_rows(scale)
                else:
                    scale_out = scale
                for tk, _ in pairs:
                    out[tk[: -len(".weight")] + ".weight_scale"] = scale_out
            if marker_key in state:
                processed_sides.add(marker_key)
                for tk, _ in pairs:
                    out[tk[: -len(".weight")] + ".comfy_quant"] = state[marker_key].clone()
            continue

        # Generic 1:1 (or multi-target via convert_transformer_key).
        if source_key not in plan and not any(
            source_key.startswith(p)
            for p in ("blocks.", "token_refiner.", "final_layer.", "time_embedder.")
        ):
            # Unknown weights are not emitted; G1 tracks them separately.
            continue

        pairs = convert_transformer_key(source_key, tensor, config)
        for tk, tv in pairs:
            out[tk] = tv

        base = source_key[: -len(".weight")]
        for suffix in (".weight_scale", ".comfy_quant", ".input_scale"):
            sk = f"{base}{suffix}"
            if sk not in state:
                continue
            processed_sides.add(sk)
            # Rename side key the same way as the weight target(s).
            # Use convert on a dummy? Prefer: map via first target rename.
            if not pairs:
                continue
            # For multi-target non-qkv (rare), attach side only if 1:1.
            if len(pairs) == 1:
                tk = pairs[0][0]
                out[tk[: -len(".weight")] + suffix] = state[sk]
            else:
                for tk, _ in pairs:
                    out[tk[: -len(".weight")] + suffix] = state[sk].clone()

    # Pass through non-weight keys that were not handled as sides of a weight.
    for key, tensor in state.items():
        if key.endswith(".weight") or key in processed_sides:
            continue
        if key in MINIMAX_H3_TRANSFORMER_DROPPED_KEYS:
            continue
        # Bias / norms without int8 sides: convert if planned.
        if key.endswith(".bias") or key.endswith(".weight") is False:
            # Treat as weight-like for rename path (bias etc.)
            if any(key.endswith(s) for s in (".weight_scale", ".comfy_quant", ".input_scale")):
                # Orphan side without matching weight in this state — skip.
                continue
            pairs = convert_transformer_key(key, tensor, config)
            for tk, tv in pairs:
                if tk not in out:
                    out[tk] = tv

    return out


def _planned_target_shapes(
    source_weight_key: str, config: dict[str, Any]
) -> list[tuple[str, list[int]]] | None:
    plan = get_transformer_key_plan(config)
    if source_weight_key not in plan:
        return None
    return plan[source_weight_key]


def g1_int8_source_coverage(
    state: dict[str, torch.Tensor],
    config: dict[str, Any],
    *,
    source_layout: SourceLayout,
) -> dict[str, Any]:
    """G1 primary ratio: mapped int8 ``.weight`` / all int8 ``.weight``."""
    state = strip_state_dict_prefixes(state)
    int8_weights = [
        k for k, t in state.items() if k.endswith(".weight") and t.dtype == torch.int8
    ]
    mapped_ok: list[str] = []
    unmapped: list[str] = []
    shape_errors: list[str] = []

    for k in int8_weights:
        planned = _planned_target_shapes(k, config)
        if planned is None or len(planned) == 0:
            unmapped.append(k)
            continue
        try:
            mini: dict[str, torch.Tensor] = {k: state[k]}
            base = k[: -len(".weight")]
            for suffix in (".weight_scale", ".comfy_quant", ".input_scale"):
                sk = f"{base}{suffix}"
                if sk in state:
                    mini[sk] = state[sk]
            converted = convert_transformer_key_with_sides(
                mini, config, source_layout=source_layout
            )
            ok = True
            for target_key, expected_shape in planned:
                if target_key not in converted:
                    shape_errors.append(f"{k} -> missing target {target_key}")
                    ok = False
                    continue
                actual = list(converted[target_key].shape)
                if actual != list(expected_shape):
                    shape_errors.append(
                        f"{k} -> {target_key}: shape {actual} != planned {expected_shape}"
                    )
                    ok = False
            if ok:
                mapped_ok.append(k)
            else:
                unmapped.append(k)
        except Exception as exc:  # noqa: BLE001 — G1 reports failures
            unmapped.append(k)
            shape_errors.append(f"{k}: {exc}")

    n = len(int8_weights)
    return {
        "int8_weight_total": n,
        "int8_weight_mapped_ok": len(mapped_ok),
        "source_coverage_ratio": len(mapped_ok) / max(n, 1),
        "unmapped_int8_weights": unmapped,
        "shape_errors": shape_errors,
        "mapped_int8_weights": mapped_ok,
    }
